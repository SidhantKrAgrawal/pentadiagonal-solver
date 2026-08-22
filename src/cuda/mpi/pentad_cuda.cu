#include <cooperative_groups.h>  // for this_grid, grid_group
#include <cassert>               // for assert
#include <cstddef>               // for size_t
#include <functional>            // for multiplies
#include <numeric>               // for accumulate
#include "pentadsolver_cuda.hpp" // for pentadsolver_gpsv_batch

namespace {
template <typename REAL>
const MPI_Datatype mpi_datatype =
    std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;
enum class communication_dir_t { DOWN = 1, UP = 2, ALL = 3 };
} // namespace

template <typename Float>
__device__ void shift_uw(const Float *ds, const Float *dl, const Float *du,
                         const Float *dw, const Float xx, size_t r1_idx,
                         Float *r0) {
  // Index naming holds for r1; for r0 it is l, d, tmp, u, w, x.
  constexpr size_t l = 1;
  constexpr size_t d = 2;
  constexpr size_t u = 3;
  constexpr size_t w = 4;
  constexpr size_t x = 5;
  Float u0_tmp       = r0[u];
  r0[x]              = r0[x] - u0_tmp * xx;
  r0[l]              = r0[l] - u0_tmp * ds[r1_idx];
  r0[d]              = r0[d] - u0_tmp * dl[r1_idx];
  r0[u]              = r0[w] - u0_tmp * du[r1_idx];
  r0[w]              = 00 - u0_tmp * dw[r1_idx];
}

template <typename Float>
__device__ void shift_sl(const Float *ds, const Float *dl, const Float *d,
                         const Float *du, const Float *dw, Float *xx,
                         Float *dss, Float *dll, Float *duu, Float *dww,
                         size_t idx, size_t t_stride_ws, size_t t_stride) {
  size_t ri      = idx * t_stride;
  size_t ri_ws   = idx * t_stride_ws;
  size_t rim1    = (idx - 1) * t_stride;
  size_t rim2    = (idx - 2) * t_stride;
  size_t rim1_ws = (idx - 1) * t_stride_ws;
  size_t rim2_ws = (idx - 2) * t_stride_ws;
  if (idx <= 3) {
    Float si   = ds[ri];
    Float li   = dl[ri];
    Float di   = d[ri] - li * duu[rim1_ws];
    xx[ri]     = (xx[ri] - li * xx[rim1]) / di;
    dss[ri_ws] = (-li * dss[rim1_ws]) / di;
    dll[ri_ws] = (si - li * dll[rim1_ws]) / di;
    duu[ri_ws] = (du[ri] - li * dww[rim1_ws]) / di;
    dww[ri_ws] = (dw[ri]) / di;
  } else {
    Float si   = ds[ri];
    Float li   = dl[ri] - si * duu[rim2_ws];
    Float di   = d[ri] - si * dww[rim2_ws] - li * duu[rim1_ws];
    xx[ri]     = (xx[ri] - si * xx[rim2] - li * xx[rim1]) / di;
    dss[ri_ws] = (-si * dss[rim2_ws] - li * dss[rim1_ws]) / di;
    dll[ri_ws] = (-si * dll[rim2_ws] - li * dll[rim1_ws]) / di;
    duu[ri_ws] = (du[ri] - li * dww[rim1_ws]) / di;
    dww[ri_ws] = (dw[ri]) / di;
  }
}

template <typename Float>
__device__ void pack_first_rows_forward(Float ds0, Float *r0, Float *r1,
                                        Float *sndbuf, size_t sys_id,
                                        size_t t_stride_ws) {
  // index namin olds for r1, in case of r0: l, d, tmp, u, w x
  constexpr size_t s    = 0;
  constexpr size_t l    = 1;
  constexpr size_t d    = 2;
  constexpr size_t u    = 3;
  constexpr size_t w    = 4;
  constexpr size_t x    = 5;
  constexpr size_t usnd = 2;
  constexpr size_t wsnd = 3;
  constexpr size_t xsnd = 4;
  // norm r1
  r1[s] = r1[s] / r1[d];
  r1[l] = r1[l] / r1[d];
  r1[u] = r1[u] / r1[d];
  r1[w] = r1[w] / r1[d];
  r1[x] = r1[x] / r1[d];
  // r0 - tmp * r1
  //  note: ds is added to the beginning of the array so indexing will change
  Float u0_tmp                               = r0[d];
  r0[d]                                      = r0[l] - u0_tmp * r1[l]; // ddi
  sndbuf[t_stride_ws * 2 * s + sys_id]       = ds0 / r0[d];
  sndbuf[t_stride_ws * 2 * l + sys_id]       = (r0[s] - u0_tmp * r1[s]) / r0[d];
  sndbuf[t_stride_ws * 2 * usnd + sys_id]    = (r0[u] - u0_tmp * r1[u]) / r0[d];
  sndbuf[t_stride_ws * 2 * wsnd + sys_id]    = (r0[w] - u0_tmp * r1[w]) / r0[d];
  sndbuf[t_stride_ws * 2 * xsnd + sys_id]    = (r0[x] - u0_tmp * r1[x]) / r0[d];
  sndbuf[t_stride_ws * (2 * s + 1) + sys_id] = r1[s];
  sndbuf[t_stride_ws * (2 * l + 1) + sys_id] = r1[l];
  sndbuf[t_stride_ws * (2 * usnd + 1) + sys_id] = r1[u];
  sndbuf[t_stride_ws * (2 * wsnd + 1) + sys_id] = r1[w];
  sndbuf[t_stride_ws * (2 * xsnd + 1) + sys_id] = r1[x];
}

template <typename Float>
__device__ void
gpsv_forward_x(const Float *__restrict__ ds, const Float *__restrict__ dl,
               const Float *__restrict__ d, const Float *__restrict__ du,
               const Float *__restrict__ dw, Float *__restrict__ x,
               Float *__restrict__ dss, Float *__restrict__ dll,
               Float *__restrict__ duu, Float *__restrict__ dww,
               Float *__restrict__ top, Float *__restrict__ bottom,
               size_t sys_idx, size_t t_stride_ws, size_t t_sys_size) {
  constexpr int n_row_nonzeros = 6;
  // So a temp value is added to the line and shifted like row 1 until the end
  // Finally du is shifted as well and ds[0] added back
  Float r0[n_row_nonzeros] = {dl[0], d[0], du[0], dw[0], 0, x[0]};
  Float r1[n_row_nonzeros] = {ds[1], dl[1], d[1], du[1], dw[1], x[1]};
  // row 2 - normalise -- 0
  x[2]                 = x[2] / d[2];
  dss[2 * t_stride_ws] = ds[2] / d[2];
  dll[2 * t_stride_ws] = dl[2] / d[2];
  duu[2 * t_stride_ws] = du[2] / d[2];
  dww[2 * t_stride_ws] = dw[2] / d[2];
  shift_uw(dss, dll, duu, dww, x[2], 2 * t_stride_ws,
           r0); // NOLINT hicpp-no-array-decay
  shift_uw(dss, dll, duu, dww, x[2], 2 * t_stride_ws,
           r1); // NOLINT hicpp-no-array-decay

  // rest
  for (size_t i = 3; i < t_sys_size; ++i) {
    // row i - (dds*row{i-2}) - ddu'-row{i-1}, normalise
    // TODO: check with Istvan -- ds, dl, du, dw requirements -- last elements
    // must be 0, but are they accessible? remove comp du2[N-1], dw2[N-2..N-1]
    shift_sl(ds, dl, d, du, dw, x, dss, dll, duu, dww, i, t_stride_ws, 1UL);
    shift_uw(dss, dll, duu, dww, x[i], i * t_stride_ws,
             r0); // NOLINT hicpp-no-array-decay
    shift_uw(dss, dll, duu, dww, x[i], i * t_stride_ws,
             r1); // NOLINT hicpp-no-array-decay
  }

  // Prepare layout of the last two row for comm
  // s l d T 0 0 ... 0 u w      s l d T 0 0 ... 0 u w
  //   s l d 0 0 ... 0 u w        s l d 0 0 ... 0 u w
  //     s l 1 u w     | |          s l 1 u w     | |
  //     s l 0 1 u w   | |          s l 0 1 u w   | |
  //     | | ...       | |          | | ...       | |
  //     s l 0 0 1 u w | |          s l 0 0 1 u w | |
  //     s l 0 0 0 1 u w |  -->     s l 0 0 0 1 0 u w
  //     s l 0 0 0 0 1 u w  -/      s l 0 0 0 0 1 u w
  size_t i             = t_sys_size - 2;
  Float li             = duu[i * t_stride_ws];
  x[i]                 = x[i] - li * x[i + 1];
  dss[i * t_stride_ws] = dss[i * t_stride_ws] - li * dss[(i + 1) * t_stride_ws];
  dll[i * t_stride_ws] = dll[i * t_stride_ws] - li * dll[(i + 1) * t_stride_ws];
  duu[i * t_stride_ws] = dww[i * t_stride_ws] - li * duu[(i + 1) * t_stride_ws];
  dww[i * t_stride_ws] = -li * dww[(i + 1) * t_stride_ws];
  // copy last two rows to combuf for reduced step
  bottom[t_stride_ws * 0 + sys_idx] = dss[i * t_stride_ws];
  bottom[t_stride_ws * 1 + sys_idx] = dss[(i + 1) * t_stride_ws];
  bottom[t_stride_ws * 2 + sys_idx] = dll[i * t_stride_ws];
  bottom[t_stride_ws * 3 + sys_idx] = dll[(i + 1) * t_stride_ws];
  bottom[t_stride_ws * 4 + sys_idx] = duu[i * t_stride_ws];
  bottom[t_stride_ws * 5 + sys_idx] = duu[(i + 1) * t_stride_ws]; // NOLINT
  bottom[t_stride_ws * 6 + sys_idx] = dww[i * t_stride_ws];       // NOLINT
  bottom[t_stride_ws * 7 + sys_idx] = dww[(i + 1) * t_stride_ws]; // NOLINT
  bottom[t_stride_ws * 8 + sys_idx] = x[i];                       // NOLINT
  bottom[t_stride_ws * 9 + sys_idx] = x[(i + 1)];                 // NOLINT
  // Prepare layout of the first two row for comm
  // NOLINTNEXTLINE hicpp-no-array-decay
  pack_first_rows_forward(ds[0], r0, r1, top, sys_idx, t_stride_ws);
}

template <typename Float>
__device__ void
gpsv_forward_strided(const Float *__restrict__ ds, const Float *__restrict__ dl,
                     const Float *__restrict__ d, const Float *__restrict__ du,
                     const Float *__restrict__ dw, Float *__restrict__ x,
                     Float *__restrict__ dss, Float *__restrict__ dll,
                     Float *__restrict__ duu, Float *__restrict__ dww,
                     Float *__restrict__ top, Float *__restrict__ bottom,
                     size_t sys_idx, size_t t_stride, size_t t_stride_ws,
                     size_t t_sys_size) {
  constexpr int n_row_nonzeros = 6;
  // So a temp value is added to the line and shifted like row 1 until the end
  // Finally du is shifted as well and ds[0] added back
  Float r0[n_row_nonzeros] = {dl[0], d[0], du[0], dw[0], 0, x[0]};
  Float r1[n_row_nonzeros] = {ds[t_stride], dl[t_stride], d[t_stride],
                              du[t_stride], dw[t_stride], x[t_stride]};
  // row 2 - normalise -- 0
  x[2 * t_stride]      = x[2 * t_stride] / d[2 * t_stride];
  dss[2 * t_stride_ws] = ds[2 * t_stride] / d[2 * t_stride];
  dll[2 * t_stride_ws] = dl[2 * t_stride] / d[2 * t_stride];
  duu[2 * t_stride_ws] = du[2 * t_stride] / d[2 * t_stride];
  dww[2 * t_stride_ws] = dw[2 * t_stride] / d[2 * t_stride];
  shift_uw(dss, dll, duu, dww, x[2 * t_stride], 2 * t_stride_ws,
           r0); // NOLINT hicpp-no-array-decay
  shift_uw(dss, dll, duu, dww, x[2 * t_stride], 2 * t_stride_ws,
           r1); // NOLINT hicpp-no-array-decay

  // rest
  for (size_t i = 3; i < t_sys_size; ++i) {
    // row i - (dds*row{i-2}) - ddu'-row{i-1}, normalise
    // TODO: check with Istvan -- ds, dl, du, dw requirements -- last elements
    // must be 0, but are they accessible? remove comp du2[N-1], dw2[N-2..N-1]
    shift_sl(ds, dl, d, du, dw, x, dss, dll, duu, dww, i, t_stride_ws,
             t_stride);
    shift_uw(dss, dll, duu, dww, x[i * t_stride], i * t_stride_ws,
             r0); // NOLINT hicpp-no-array-decay
    shift_uw(dss, dll, duu, dww, x[i * t_stride], i * t_stride_ws,
             r1); // NOLINT hicpp-no-array-decay
  }

  // Prepare layout of the last two row for comm
  // s l d T 0 0 ... 0 u w      s l d T 0 0 ... 0 u w
  //   s l d 0 0 ... 0 u w        s l d 0 0 ... 0 u w
  //     s l 1 u w     | |          s l 1 u w     | |
  //     s l 0 1 u w   | |          s l 0 1 u w   | |
  //     | | ...       | |          | | ...       | |
  //     s l 0 0 1 u w | |          s l 0 0 1 u w | |
  //     s l 0 0 0 1 u w |  -->     s l 0 0 0 1 0 u w
  //     s l 0 0 0 0 1 u w  -/      s l 0 0 0 0 1 u w
  size_t i             = t_sys_size - 2;
  Float li             = duu[i * t_stride_ws];
  x[i * t_stride]      = x[i * t_stride] - li * x[(i + 1) * t_stride];
  dss[i * t_stride_ws] = dss[i * t_stride_ws] - li * dss[(i + 1) * t_stride_ws];
  dll[i * t_stride_ws] = dll[i * t_stride_ws] - li * dll[(i + 1) * t_stride_ws];
  duu[i * t_stride_ws] = dww[i * t_stride_ws] - li * duu[(i + 1) * t_stride_ws];
  dww[i * t_stride_ws] = -li * dww[(i + 1) * t_stride_ws];
  // copy last two rows to combuf for reduced step
  bottom[t_stride_ws * 0 + sys_idx] = dss[i * t_stride_ws];
  bottom[t_stride_ws * 1 + sys_idx] = dss[(i + 1) * t_stride_ws];
  bottom[t_stride_ws * 2 + sys_idx] = dll[i * t_stride_ws];
  bottom[t_stride_ws * 3 + sys_idx] = dll[(i + 1) * t_stride_ws];
  bottom[t_stride_ws * 4 + sys_idx] = duu[i * t_stride_ws];
  bottom[t_stride_ws * 5 + sys_idx] = duu[(i + 1) * t_stride_ws]; // NOLINT
  bottom[t_stride_ws * 6 + sys_idx] = dww[i * t_stride_ws];       // NOLINT
  bottom[t_stride_ws * 7 + sys_idx] = dww[(i + 1) * t_stride_ws]; // NOLINT
  bottom[t_stride_ws * 8 + sys_idx] = x[i * t_stride];            // NOLINT
  bottom[t_stride_ws * 9 + sys_idx] = x[(i + 1) * t_stride];      // NOLINT
  // Prepare layout of the first two row for comm
  // NOLINTNEXTLINE hicpp-no-array-decay
  pack_first_rows_forward(ds[0], r0, r1, top, sys_idx, t_stride_ws);
}

template <typename Float>
__global__ void gpsv_forward_batch_x_kernel(
    const Float *__restrict__ ds, const Float *__restrict__ dl,
    const Float *__restrict__ d, const Float *__restrict__ du,
    const Float *__restrict__ dw, Float *__restrict__ x,
    Float *__restrict__ dss, Float *__restrict__ dll, Float *__restrict__ duu,
    Float *__restrict__ dww, Float *__restrict__ top,
    Float *__restrict__ bottom, size_t t_n_sys, size_t t_sys_size) {
  size_t tid = cooperative_groups::this_grid().thread_rank();
  if (tid < t_n_sys) {
    size_t start_idx = tid * t_sys_size;
    // buffers use coalesced accessses
    size_t ws_start_idx = tid;
    gpsv_forward_x(ds + start_idx, dl + start_idx, d + start_idx,
                   du + start_idx, dw + start_idx, x + start_idx,
                   dss + ws_start_idx, dll + ws_start_idx, duu + ws_start_idx,
                   dww + ws_start_idx, top, bottom, tid, t_n_sys, t_sys_size);
  }
}

template <typename Float>
__global__ void gpsv_forward_batch_outermost_kernel(
    const Float *__restrict__ ds, const Float *__restrict__ dl,
    const Float *__restrict__ d, const Float *__restrict__ du,
    const Float *__restrict__ dw, Float *__restrict__ x,
    Float *__restrict__ dss, Float *__restrict__ dll, Float *__restrict__ duu,
    Float *__restrict__ dww, Float *__restrict__ top,
    Float *__restrict__ bottom, size_t t_n_sys, size_t t_sys_size) {
  size_t tid = cooperative_groups::this_grid().thread_rank();
  if (tid < t_n_sys) {
    size_t start_idx = tid;
    // buffers use coalesced accessses
    size_t ws_start_idx = tid;
    gpsv_forward_strided(ds + start_idx, dl + start_idx, d + start_idx,
                         du + start_idx, dw + start_idx, x + start_idx,
                         dss + ws_start_idx, dll + ws_start_idx,
                         duu + ws_start_idx, dww + ws_start_idx, top, bottom,
                         tid, t_n_sys, t_n_sys, t_sys_size);
  }
}

template <typename Float>
__global__ void gpsv_forward_batch_middle_kernel(
    const Float *__restrict__ ds, const Float *__restrict__ dl,
    const Float *__restrict__ d, const Float *__restrict__ du,
    const Float *__restrict__ dw, Float *__restrict__ x,
    Float *__restrict__ dss, Float *__restrict__ dll, Float *__restrict__ duu,
    Float *__restrict__ dww, Float *__restrict__ top,
    Float *__restrict__ bottom, size_t t_n_sys_in, size_t t_n_sys_out,
    size_t t_sys_size) {
  size_t tid     = cooperative_groups::this_grid().thread_rank();
  size_t t_n_sys = t_n_sys_in * t_n_sys_out;
  if (tid < t_n_sys) {
    int page_idx     = tid / t_n_sys_in;
    int sys_in_page  = tid % t_n_sys_in;
    size_t start_idx = page_idx * t_n_sys_in * t_sys_size + sys_in_page;
    // buffers use coalesced accessses
    size_t ws_start_idx = tid;
    gpsv_forward_strided(ds + start_idx, dl + start_idx, d + start_idx,
                         du + start_idx, dw + start_idx, x + start_idx,
                         dss + ws_start_idx, dll + ws_start_idx,
                         duu + ws_start_idx, dww + ws_start_idx, top, bottom,
                         tid, t_n_sys_in, t_n_sys, t_sys_size);
  }
}

template <typename Float>
void gpsv_batched_forward(const Float *ds, const Float *dl, const Float *d,
                          const Float *du, const Float *dw, Float *x,
                          Float *dss, Float *dll, Float *duu, Float *dww,
                          Float *comm_buf, Float *snd_bottom_buf,
                          const int *t_dims, size_t t_ndims, size_t t_n_sys,
                          int t_solvedim) {
  constexpr int block_dim_x = 128;
  int nblocks               = 1 + (static_cast<int>(t_n_sys) - 1) / block_dim_x;
  if (t_solvedim == 0) {
    gpsv_forward_batch_x_kernel<<<block_dim_x, nblocks>>>(
        ds, dl, d, du, dw, x, dss, dll, duu, dww, comm_buf, snd_bottom_buf,
        t_n_sys, t_dims[t_solvedim]);
  } else if (t_solvedim == t_ndims - 1) {
    gpsv_forward_batch_outermost_kernel<<<block_dim_x, nblocks>>>(
        ds, dl, d, du, dw, x, dss, dll, duu, dww, comm_buf, snd_bottom_buf,
        t_n_sys, t_dims[t_solvedim]);
  } else {
    size_t n_sys_in =
        std::accumulate(t_dims, t_dims + t_solvedim, 1, std::multiplies<>());
    size_t n_sys_out = std::accumulate(
        t_dims + t_solvedim + 1, t_dims + t_ndims, 1, std::multiplies<>());
    gpsv_forward_batch_middle_kernel<<<block_dim_x, nblocks>>>(
        ds, dl, d, du, dw, x, dss, dll, duu, dww, comm_buf, snd_bottom_buf,
        n_sys_in, n_sys_out, t_dims[t_solvedim]);
  }
}
// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------

template <typename Float>
__device__ void gpsv_backward_x(const Float *dss, const Float *dll,
                                const Float *duu, const Float *dww, Float *x,
                                Float x0, Float x1, Float xp1, Float xp2,
                                int t_stride_ws, int t_sys_size) {
  x[0] = x0;
  x[1] = x1;
  // last two rows:
  {
    int i = t_sys_size - 1;
    x[i]  = x[i] - dww[i * t_stride_ws] * xp2 - duu[i * t_stride_ws] * xp1 -
           dll[i * t_stride_ws] * x1 - dss[i * t_stride_ws] * x0;
    i    = t_sys_size - 2;
    x[i] = x[i] - dww[i * t_stride_ws] * xp2 - duu[i * t_stride_ws] * xp1 -
           dll[i * t_stride_ws] * x1 - dss[i * t_stride_ws] * x0;
    xp1 = x[i];
    xp2 = x[i + 1];
  }

  for (int i = t_sys_size - 3; i > 1; --i) {
    x[i] = x[i] - dww[i * t_stride_ws] * xp2 - duu[i * t_stride_ws] * xp1 -
           dll[i * t_stride_ws] * x1 - dss[i * t_stride_ws] * x0;
    xp2 = xp1;
    xp1 = x[i];
  }
}

template <typename Float>
__device__ void gpsv_backward_strided(const Float *dss, const Float *dll,
                                      const Float *duu, const Float *dww,
                                      Float *x, Float x0, Float x1, Float xp1,
                                      Float xp2, int t_stride, int t_stride_ws,
                                      int t_sys_size) {
  x[0]            = x0;
  x[1 * t_stride] = x1;
  // last two rows:
  {
    int i           = t_sys_size - 1;
    x[i * t_stride] = x[i * t_stride] - dww[i * t_stride_ws] * xp2 -
                      duu[i * t_stride_ws] * xp1 - dll[i * t_stride_ws] * x1 -
                      dss[i * t_stride_ws] * x0;
    i               = t_sys_size - 2;
    x[i * t_stride] = x[i * t_stride] - dww[i * t_stride_ws] * xp2 -
                      duu[i * t_stride_ws] * xp1 - dll[i * t_stride_ws] * x1 -
                      dss[i * t_stride_ws] * x0;
    xp1 = x[i * t_stride];
    xp2 = x[(i + 1) * t_stride];
  }

  for (int i = t_sys_size - 3; i > 1; --i) {
    x[i * t_stride] = x[i * t_stride] - dww[i * t_stride_ws] * xp2 -
                      duu[i * t_stride_ws] * xp1 - dll[i * t_stride_ws] * x1 -
                      dss[i * t_stride_ws] * x0;
    xp2 = xp1;
    xp1 = x[i * t_stride];
  }
}

template <typename Float>
__global__ void gpsv_backward_batch_x(const Float *dss, const Float *dll,
                                      const Float *duu, const Float *dww,
                                      Float *x, const Float *x_top,
                                      const Float *xp1, int t_n_sys,
                                      int t_stride_ws, int t_sys_size) {
  size_t tid = cooperative_groups::this_grid().thread_rank();
  if (tid < t_n_sys) {
    size_t sys_start = tid * t_sys_size;
    size_t ws0       = tid;
    size_t ws1       = tid + t_stride_ws;

    gpsv_backward_x(dss + ws0, dll + ws0, duu + ws0, dww + ws0, x + sys_start,
                    x_top[ws0], x_top[ws1], xp1[ws0], xp1[ws1], t_stride_ws,
                    t_sys_size);
  }
}

template <typename Float>
__global__ void
gpsv_backward_batch_outermost(const Float *dss, const Float *dll,
                              const Float *duu, const Float *dww, Float *x,
                              const Float *x_top, const Float *xp1, int t_n_sys,
                              int t_sys_size) {
  size_t tid = cooperative_groups::this_grid().thread_rank();
  if (tid < t_n_sys) {
    size_t sys_start = tid;
    size_t ws0       = tid;
    size_t ws1       = tid + t_n_sys;
    gpsv_backward_strided(dss + sys_start, dll + sys_start, duu + sys_start,
                          dww + sys_start, x + sys_start, x_top[ws0],
                          x_top[ws1], xp1[ws0], xp1[ws1], t_n_sys, t_n_sys,
                          t_sys_size);
  }
}

template <typename Float>
__global__ void
gpsv_backward_batch_middle(const Float *dss, const Float *dll, const Float *duu,
                           const Float *dww, Float *x, const Float *x_top,
                           const Float *xp1, int t_n_sys_in, int t_n_sys_out,
                           int t_stride_ws, int t_sys_size) {
  size_t tid  = cooperative_groups::this_grid().thread_rank();
  int t_n_sys = t_n_sys_in * t_n_sys_out;
  if (tid < t_n_sys) {
    int page_idx     = tid / t_n_sys_in;
    int sys_in_page  = tid % t_n_sys_in;
    size_t sys_start = page_idx * t_n_sys_in * t_sys_size + sys_in_page;
    size_t ws0       = tid;
    size_t ws1       = tid + t_stride_ws;
    gpsv_backward_strided(dss + ws0, dll + ws0, duu + ws0, dww + ws0,
                          x + sys_start, x_top[ws0], x_top[ws1], xp1[ws0],
                          xp1[ws1], t_n_sys_in, t_n_sys, t_sys_size);
  }
}

template <typename Float>
void gpsv_batched_backward(const Float *dss, const Float *dll, const Float *duu,
                           const Float *dww, Float *x, const Float *x_top,
                           const Float *xp1, const int *t_dims, int t_ndims,
                           int t_n_sys, int t_solvedim) {
  constexpr int block_dim_x = 128;
  int nblocks               = 1 + (static_cast<int>(t_n_sys) - 1) / block_dim_x;
  if (t_solvedim == 0) {
    gpsv_backward_batch_x<<<block_dim_x, nblocks>>>(
        dss, dll, duu, dww, x, x_top, xp1, t_n_sys, t_n_sys,
        t_dims[t_solvedim]);
  } else if (t_solvedim == t_ndims - 1) {
    gpsv_backward_batch_outermost<<<block_dim_x, nblocks>>>(
        dss, dll, duu, dww, x, x_top, xp1, t_n_sys, t_dims[t_solvedim]);
  } else {
    size_t n_sys_in =
        std::accumulate(t_dims, t_dims + t_solvedim, 1, std::multiplies<>());
    size_t n_sys_out = std::accumulate(
        t_dims + t_solvedim + 1, t_dims + t_ndims, 1, std::multiplies<>());
    gpsv_backward_batch_middle<<<block_dim_x, nblocks>>>(
        dss, dll, duu, dww, x, x_top, xp1, n_sys_in, n_sys_out, t_n_sys,
        t_dims[t_solvedim]);
  }
}

// ----------------------------------------------------------------------------
// Reduced Solve
// ----------------------------------------------------------------------------
template <communication_dir_t dir, typename Float>
void send_rows_to_nodes(pentadsolver_handle_t params, int t_solvedim,
                        int distance, int msg_size, const Float *sndbuf,
                        Float *rcvbufL, Float *rcvbufR, int tag = 1242) {
  int rank                        = params->mpi_coords[t_solvedim];
  int nproc                       = params->num_mpi_procs[t_solvedim];
  int leftrank                    = rank - distance;
  int rightrank                   = rank + distance;
  std::array<MPI_Request, 4> reqs = {
      MPI_REQUEST_NULL,
      MPI_REQUEST_NULL,
      MPI_REQUEST_NULL,
      MPI_REQUEST_NULL,
  };
  // Get the minus elements
  if (leftrank >= 0) {
    // send recv
    if (static_cast<unsigned>(dir) &
        static_cast<unsigned>(communication_dir_t::UP)) {
      MPI_Isend(sndbuf, msg_size, mpi_datatype<Float>, leftrank, tag,
                params->communicators[t_solvedim], &reqs[2]);
    }
    if (static_cast<unsigned>(dir) &
        static_cast<unsigned>(communication_dir_t::DOWN)) {
      MPI_Irecv(rcvbufL, msg_size, mpi_datatype<Float>, leftrank, tag,
                params->communicators[t_solvedim], &reqs[0]);
    }
  }

  // Get the plus elements
  if (rightrank < nproc) {
    // send recv
    if (static_cast<unsigned>(dir) &
        static_cast<unsigned>(communication_dir_t::DOWN)) {
      MPI_Isend(sndbuf, msg_size, mpi_datatype<Float>, rightrank, tag,
                params->communicators[t_solvedim], &reqs[3]);
    }
    if (static_cast<unsigned>(dir) &
        static_cast<unsigned>(communication_dir_t::UP)) {
      MPI_Irecv(rcvbufR, msg_size, mpi_datatype<Float>, rightrank, tag,
                params->communicators[t_solvedim], &reqs[1]);
    }
  }

  // Wait for receives to finish
  MPI_Waitall(4, reqs.data(), MPI_STATUS_IGNORE);
}

template <communication_dir_t dir, typename Float>
void send_rows_to_nodes(pentadsolver_handle_t params, int t_solvedim,
                        int distance, int msg_size, const Float *sndbuf_d,
                        Float *sndbuf_h, Float *rcvbufL_d, Float *rcvbufL_h,
                        Float *rcvbufR_d, Float *rcvbufR_h, int tag = 1242) {
  int rank      = params->mpi_coords[t_solvedim];
  int nproc     = params->num_mpi_procs[t_solvedim];
  int leftrank  = rank - distance;
  int rightrank = rank + distance;
  if (leftrank >= 0 || rightrank < nproc) {
    cudaMemcpy(sndbuf_h, sndbuf_d, msg_size * sizeof(Float),
               cudaMemcpyDeviceToHost);
  }
  send_rows_to_nodes<dir, Float>(params, t_solvedim, distance, msg_size,
                                 sndbuf_h, rcvbufL_h, rcvbufR_h, tag);
  if (leftrank >= 0 && (static_cast<unsigned>(dir) &
                        static_cast<unsigned>(communication_dir_t::DOWN))) {

    cudaMemcpy(rcvbufL_d, rcvbufL_h, msg_size * sizeof(Float),
               cudaMemcpyHostToDevice);
  }
  if (rightrank < nproc && (static_cast<unsigned>(dir) &
                            static_cast<unsigned>(communication_dir_t::UP))) {
    cudaMemcpy(rcvbufR_d, rcvbufR_h, msg_size * sizeof(Float),
               cudaMemcpyHostToDevice);
  }
}

template <typename Float>
__device__ void
shift_sl_reduced(const Float *rmds, const Float *rmdl, const Float *rmdu,
                 const Float *rmdw, const Float *rmx, Float *dss, Float *dll,
                 Float *duu, Float *dww, Float *xx, size_t t_stride, int rank) {
  Float r1l_tmp = dll[t_stride];
  dll[t_stride] = 0.0;

  if (rank > 0) {
    // row 1
    Float si      = dss[t_stride];
    Float di_tmp  = 1 - si * rmdw[t_stride];
    r1l_tmp       = (r1l_tmp - si * rmdu[t_stride]) / di_tmp;
    dss[t_stride] = (-si * rmds[t_stride]) / di_tmp;
    dll[t_stride] = (-si * rmdl[t_stride]) / di_tmp;
    duu[t_stride] = (duu[t_stride]) / di_tmp;
    dww[t_stride] = (dww[t_stride]) / di_tmp;
    xx[t_stride]  = (xx[t_stride] - si * rmx[t_stride]) / di_tmp;
    // row 0
    si            = dss[0];
    Float li      = dll[0];
    Float r0u_tmp = -si * rmdw[0] - li * rmdw[t_stride];
    di_tmp        = 1 - si * rmdu[0] - li * rmdu[t_stride] - r0u_tmp * r1l_tmp;
    dss[0] = (-si * rmds[0] - li * rmds[t_stride] - r0u_tmp * dss[t_stride]) /
             di_tmp;
    dll[0] = (-si * rmdl[0] - li * rmdl[t_stride] - r0u_tmp * dll[t_stride]) /
             di_tmp;
    duu[0] = (duu[0] - r0u_tmp * duu[t_stride]) / di_tmp;
    dww[0] = (dww[0] - r0u_tmp * dww[t_stride]) / di_tmp;
    xx[0] =
        (xx[0] - si * rmx[0] - li * rmx[t_stride] - r0u_tmp * xx[t_stride]) /
        di_tmp;
  } else {
    // align indexing of s, l in row 1 to row 0
    dll[t_stride] = dss[t_stride];
    dss[t_stride] = 0.0;
  }
  dss[t_stride] = dss[t_stride] - r1l_tmp * dss[0];
  dll[t_stride] = dll[t_stride] - r1l_tmp * dll[0];
  duu[t_stride] = duu[t_stride] - r1l_tmp * duu[0];
  dww[t_stride] = dww[t_stride] - r1l_tmp * dww[0];
  xx[t_stride]  = xx[t_stride] - r1l_tmp * xx[0];
}

template <typename Float>
__global__ void eliminate_bottom_row_kernel(int rank, int t_n_sys,
                                            const Float *rmi, Float *top_rows) {
  size_t tid = cooperative_groups::this_grid().thread_rank();
  if (tid < t_n_sys) {
    // buffers use coalesced accessses
    size_t start_idx  = tid;
    Float *tds        = top_rows + t_n_sys * 0 + start_idx;
    Float *tdl        = top_rows + t_n_sys * 2 + start_idx;
    Float *tdu        = top_rows + t_n_sys * 4 + start_idx;
    Float *tdw        = top_rows + t_n_sys * 6 + start_idx;
    Float *tx         = top_rows + t_n_sys * 8 + start_idx;
    const Float *rmds = rmi + t_n_sys * 0 + start_idx;
    const Float *rmdl = rmi + t_n_sys * 2 + start_idx;
    const Float *rmdu = rmi + t_n_sys * 4 + start_idx;
    const Float *rmdw = rmi + t_n_sys * 6 + start_idx;
    const Float *rmx  = rmi + t_n_sys * 8 + start_idx;

    shift_sl_reduced(rmds, rmdl, rmdu, rmdw, rmx, tds, tdl, tdu, tdw, tx,
                     t_n_sys, rank);
  }
}

template <typename Float>
void eliminate_bottom_rows_from_reduced(pentadsolver_handle_t params,
                                        int t_solvedim, int t_n_sys,
                                        const Float *snd_bottom_buf_d,
                                        Float *snd_bottom_buf_h,
                                        Float *rcvbuf_d, Float *rcvbuf_h,
                                        Float *top_buf_d, Float *top_buf_h) {
  int rank                   = params->mpi_coords[t_solvedim];
  constexpr int nvar_per_sys = 10;
  send_rows_to_nodes<communication_dir_t::DOWN, Float>(
      params, t_solvedim, 1, t_n_sys * nvar_per_sys, snd_bottom_buf_d,
      snd_bottom_buf_h, rcvbuf_d, rcvbuf_h, nullptr, nullptr);
  // s l 1 0 u w         -> s l 1 0 u w
  // s l 0 1 u w         -> s l 0 1 u w
  // ------------------- -> ---------------
  //     s l 1 0 u w     -> s l     1 0 u w
  //       s l 1 u w     -> s l     0 1 u w
  // ------------------- -> ---------------
  constexpr int block_dim_x = 128;
  int nblocks               = 1 + (static_cast<int>(t_n_sys) - 1) / block_dim_x;

  eliminate_bottom_row_kernel<<<block_dim_x, nblocks>>>(rank, t_n_sys, rcvbuf_d,
                                                        top_buf_d);
}

template <typename Float>
__device__ void pcr_iteration(
    const Float *rmds, const Float *rmdl, const Float *rmdu, const Float *rmdw,
    const Float *rmx, Float *dss, Float *dll, Float *duu, Float *dww, Float *xx,
    const Float *rpds, const Float *rpdl, const Float *rpdu, const Float *rpdw,
    const Float *rpx, size_t t_stride, int leftrank, int rightrank, int nproc) {
  // s l 1 0 u w         -> s l 1 0 u w
  // s l 0 1 u w         -> s l 0 1 u w
  // ------------------- -> -------------------
  //     s l 1 0 u w     -> s l     1 0     u w
  //     s l 0 1 u w     -> s l     0 1     u w
  // ------------------- -> -------------------
  //         s l 1 0 u w ->         s l 1 0 u w
  //         s l 0 1 u w ->         s l 0 1 u w
  constexpr size_t nvar_per_row = 5;
  constexpr size_t s            = 0;
  constexpr size_t l            = 1;
  constexpr size_t u            = 2;
  constexpr size_t w            = 3;
  constexpr size_t x            = 4;

  Float rm2[nvar_per_row] = {};
  Float rm1[nvar_per_row] = {};
  Float rp1[nvar_per_row] = {};
  Float rp2[nvar_per_row] = {};
  if (leftrank >= 0) {
    rm2[0] = rmds[0];
    rm1[0] = rmds[t_stride];
    rm2[1] = rmdl[0];
    rm1[1] = rmdl[t_stride];
    rm2[2] = rmdu[0];
    rm1[2] = rmdu[t_stride];
    rm2[3] = rmdw[0];
    rm1[3] = rmdw[t_stride];
    rm2[4] = rmx[0];
    rm1[4] = rmx[t_stride];
  }
  if (rightrank < nproc) {
    rp1[0] = rpds[0];
    rp2[0] = rpds[t_stride];
    rp1[1] = rpdl[0];
    rp2[1] = rpdl[t_stride];
    rp1[2] = rpdu[0];
    rp2[2] = rpdu[t_stride];
    rp1[3] = rpdw[0];
    rp2[3] = rpdw[t_stride];
    rp1[4] = rpx[0];
    rp2[4] = rpx[t_stride];
  }
  Float r0[nvar_per_row] = {dss[0], dll[0], duu[0], dww[0], xx[0]};
  Float r1[nvar_per_row] = {dss[t_stride], dll[t_stride], duu[t_stride],
                            dww[t_stride], xx[t_stride]};

  // shift by one:
  Float d0 =
      1 - r0[l] * rm1[u] - r0[s] * rm2[u] - r0[u] * rp1[s] - r0[w] * rp2[s];
  Float tmp0 =
      -r0[l] * rm1[w] - r0[s] * rm2[w] - r0[u] * rp1[l] - r0[w] * rp2[l];
  Float d1 =
      1 - r1[l] * rm1[w] - r1[s] * rm2[w] - r1[u] * rp1[l] - r1[w] * rp2[l];
  Float tmp1 =
      -r1[l] * rm1[u] - r1[s] * rm2[u] - r1[u] * rp1[s] - r1[w] * rp2[s];
  dss[0] = -r0[l] * rm1[s] - r0[s] * rm2[s];
  dll[0] = -r0[l] * rm1[l] - r0[s] * rm2[l];
  duu[0] = -r0[u] * rp1[u] - r0[w] * rp2[u];
  dww[0] = -r0[u] * rp1[w] - r0[w] * rp2[w];
  xx[0] =
      r0[x] - r0[u] * rp1[x] - r0[w] * rp2[x] - r0[l] * rm1[x] - r0[s] * rm2[x];
  dss[t_stride] = -r1[l] * rm1[s] - r1[s] * rm2[s];
  dll[t_stride] = -r1[l] * rm1[l] - r1[s] * rm2[l];
  duu[t_stride] = -r1[u] * rp1[u] - r1[w] * rp2[u];
  dww[t_stride] = -r1[u] * rp1[w] - r1[w] * rp2[w];
  xx[t_stride] =
      r1[x] - r1[u] * rp1[x] - r1[w] * rp2[x] - r1[l] * rm1[x] - r1[s] * rm2[x];
  // zero out tmp values
  //     s l d T u w     -> s l 1 0 u w
  //     s l T d u w     -> s l T d u w
  Float coeff = tmp0 / d1;
  d0          = d0 - coeff * tmp1;
  dss[0]      = (dss[0] - dss[t_stride] * coeff) / d0;
  dll[0]      = (dll[0] - dll[t_stride] * coeff) / d0;
  duu[0]      = (duu[0] - duu[t_stride] * coeff) / d0;
  dww[0]      = (dww[0] - dww[t_stride] * coeff) / d0;
  xx[0]       = (xx[0] - xx[t_stride] * coeff) / d0;
  //     s l 1 0 u w     -> s l 1 0 u w
  //     s l T d u w     -> s l 0 1 u w
  dss[t_stride] = (dss[t_stride] - dss[0] * tmp1) / d1;
  dll[t_stride] = (dll[t_stride] - dll[0] * tmp1) / d1;
  duu[t_stride] = (duu[t_stride] - duu[0] * tmp1) / d1;
  dww[t_stride] = (dww[t_stride] - dww[0] * tmp1) / d1;
  xx[t_stride]  = (xx[t_stride] - xx[0] * tmp1) / d1;
}

template <typename Float>
__global__ void pcr_iteration_kernel(int leftrank, int rightrank, int nproc,
                                     int t_n_sys, const Float *rpi,
                                     const Float *rmi, Float *ri) {
  size_t tid = cooperative_groups::this_grid().thread_rank();
  if (tid < t_n_sys) {
    // buffers use coalesced accessses
    size_t start_idx  = tid;
    Float *tds        = ri + t_n_sys * 0 + start_idx;
    Float *tdl        = ri + t_n_sys * 2 + start_idx;
    Float *tdu        = ri + t_n_sys * 4 + start_idx;
    Float *tdw        = ri + t_n_sys * 6 + start_idx;
    Float *tx         = ri + t_n_sys * 8 + start_idx;
    const Float *rmds = rmi + t_n_sys * 0 + start_idx;
    const Float *rmdl = rmi + t_n_sys * 2 + start_idx;
    const Float *rmdu = rmi + t_n_sys * 4 + start_idx;
    const Float *rmdw = rmi + t_n_sys * 6 + start_idx;
    const Float *rmx  = rmi + t_n_sys * 8 + start_idx;
    const Float *rpds = rpi + t_n_sys * 0 + start_idx;
    const Float *rpdl = rpi + t_n_sys * 2 + start_idx;
    const Float *rpdu = rpi + t_n_sys * 4 + start_idx;
    const Float *rpdw = rpi + t_n_sys * 6 + start_idx;
    const Float *rpx  = rpi + t_n_sys * 8 + start_idx;
    pcr_iteration(rmds, rmdl, rmdu, rmdw, rmx, tds, tdl, tdu, tdw, tx, rpds,
                  rpdl, rpdu, rpdw, rpx, t_n_sys, leftrank, rightrank, nproc);
  }
}
// Solve reduced system with PCR algorithm.
// sndbuf holds first two row of each system after the forward pass
// as an input and the solutions at the x positions as an output.
// rcvbuf must be an array with size at least 2 * 10 * n_sys.
// rcvbuf_d[0:nsys], rcvbuf_d[nsys+1:2*nsys] will hold the solution
// xp1, xp2 on exit, respectively
template <typename Float>
inline void solve_reduced_pcr(pentadsolver_handle_t params, Float *rcvbuf_d,
                              Float *sndbuf_d, Float *rcvbuf_h, Float *sndbuf_h,
                              int t_solvedim, int t_n_sys) {
  int rank                  = params->mpi_coords[t_solvedim];
  int nproc                 = params->num_mpi_procs[t_solvedim];
  constexpr int block_dim_x = 128;
  int nblocks               = 1 + (static_cast<int>(t_n_sys) - 1) / block_dim_x;
  constexpr int nvar_per_sys = 10;
  Float *rcvbufL_h           = rcvbuf_h;
  Float *rcvbufL_d           = rcvbuf_d;
  // eliminate_bottom_rows require whole systems in messages
  Float *rcvbufR_h = rcvbuf_h + t_n_sys * nvar_per_sys;
  Float *rcvbufR_d = rcvbuf_d + t_n_sys * nvar_per_sys;

  // eliminate 2 rows from reduced system
  eliminate_bottom_rows_from_reduced(params, t_solvedim, t_n_sys, rcvbufL_d,
                                     rcvbufL_h, rcvbufR_d, rcvbufR_h, sndbuf_d,
                                     sndbuf_h);

  int P    = std::ceil(std::log2((double)nproc));
  int dist = 1;
  // perform pcr
  // loop for for comm
  for (int p = 0; p < P; ++p) {
    send_rows_to_nodes<communication_dir_t::ALL, Float>(
        params, t_solvedim, dist, t_n_sys * nvar_per_sys, sndbuf_d, sndbuf_h,
        rcvbufL_d, rcvbufL_h, rcvbufR_d, rcvbufR_h);
    int leftrank  = rank - dist;
    int rightrank = rank + dist;

    // PCR algorithm
    pcr_iteration_kernel<<<block_dim_x, nblocks>>>(
        leftrank, rightrank, nproc, t_n_sys, rcvbufR_d, rcvbufL_d, sndbuf_d);

    // done
    dist = dist << 1; // NOLINT
  }
  // send solution to rank - 1 (left/UP)
  send_rows_to_nodes<communication_dir_t::UP, Float>(
      params, t_solvedim, 1, t_n_sys * 2, sndbuf_d + t_n_sys * 2 * 4, sndbuf_h,
      nullptr, nullptr, rcvbufR_d, rcvbufR_h);
}

template <typename Float>
__device__ void jacobi_iteration(const Float *rmx, const Float *s,
                                 const Float *l, const Float *u, const Float *w,
                                 const Float *xx,
                                 Float *x_cur,
                                 const Float *rpx, size_t t_stride,
                                 int leftrank, int rightrank, int nproc) {
  x_cur[0]        = xx[0];
  x_cur[t_stride] = xx[t_stride];
  if (rightrank < nproc) {
    Float xp1 = rpx[0];
    Float xp2 = rpx[t_stride];
    x_cur[0] -= u[0] * xp1 + w[0] * xp2;
    x_cur[t_stride] -= u[t_stride] * xp1 + w[t_stride] * xp2;
  }
  if (leftrank >= 0) {
    Float xm2 = rmx[0];
    Float xm1 = rmx[t_stride];
    x_cur[0] -= s[0] * xm2 + l[0] * xm1;
    x_cur[t_stride] -= s[t_stride] * xm2 + l[t_stride] * xm1;
  }
}
template <typename Float>
__global__ void jacobi_iteration_kernel(int leftrank, int rightrank, int nproc,
                                        int t_n_sys, const Float *rpi,
                                        const Float *rmi, const Float *ri,
                                        Float *x_cur) {
  size_t tid = cooperative_groups::this_grid().thread_rank();
  if (tid < t_n_sys) {
    // buffers use coalesced accessses
    size_t start_idx = tid;
    const Float *tds = ri + t_n_sys * 0 + start_idx;
    const Float *tdl = ri + t_n_sys * 2 + start_idx;
    const Float *tdu = ri + t_n_sys * 4 + start_idx;
    const Float *tdw = ri + t_n_sys * 6 + start_idx;
    const Float *tx  = ri + t_n_sys * 8 + start_idx;
    Float *x         = x_cur + start_idx;
    const Float *rmx = rmi + start_idx;
    const Float *rpx = rpi + start_idx;
    jacobi_iteration(rmx, tds, tdl, tdu, tdw, tx, x_cur, rpx, t_n_sys, leftrank,
                     rightrank, nproc);
  }
}

template <typename Float>
inline void solve_reduced_jacobi(pentadsolver_handle_t params, Float *rcvbuf_d,
                                 Float *sndbuf_d, Float *rcvbuf_h,
                                 Float *sndbuf_h, int t_solvedim, int t_n_sys) {
  int rank                  = params->mpi_coords[t_solvedim];
  int nproc                 = params->num_mpi_procs[t_solvedim];
  constexpr int block_dim_x = 128;
  int nblocks               = 1 + (static_cast<int>(t_n_sys) - 1) / block_dim_x;
  constexpr int nvar_per_sys = 10;
  Float *rcvbufL_h           = rcvbuf_h;
  Float *rcvbufL_d           = rcvbuf_d;
  // eliminate_bottom_rows require whole systems in messages
  Float *rcvbufR_h = rcvbuf_h + t_n_sys * nvar_per_sys;
  Float *rcvbufR_d = rcvbuf_d + t_n_sys * nvar_per_sys;

  // eliminate 2 rows from reduced system
  eliminate_bottom_rows_from_reduced(params, t_solvedim, t_n_sys, rcvbufL_d,
                                     rcvbufL_h, rcvbufR_d, rcvbufR_h, sndbuf_d,
                                     sndbuf_h);
  Float *x_cur_h  = rcvbuf_h;
  Float *x_prev_h = x_cur_h + 2 * t_n_sys;
  rcvbufL_h       = x_prev_h + 2 * t_n_sys;
  rcvbufR_h       = rcvbufL_h + 2 * t_n_sys;
  Float *x_cur_d  = rcvbuf_d;
  Float *x_prev_d = x_cur_d + 2 * t_n_sys;
  rcvbufL_d       = x_prev_d + 2 * t_n_sys;
  rcvbufR_d       = rcvbufL_d + 2 * t_n_sys;

  int iter                     = 0;
  constexpr int max_iter       = 100;   // TODO move to params
  constexpr double jacobi_atol = 1e-12; // TODO move to params
  constexpr double jacobi_rtol = 1e-11; // TODO move to params
  MPI_Request norm_req         = MPI_REQUEST_NULL;

  double local_norm       = -1;
  double local_norm_send  = -1;
  double global_norm0     = -1;
  double global_norm_recv = -1;
  bool need_iter          = true;

  do {
    send_rows_to_nodes<communication_dir_t::ALL, Float>(
        params, t_solvedim, 1, t_n_sys * nvar_per_sys, sndbuf_d, sndbuf_h,
        rcvbufL_d, rcvbufL_h, rcvbufR_d, rcvbufR_h);
    // kernel
    jacobi_iteration_kernel<<<block_dim_x, nblocks>>>(
        rank - 1, rank + 1, nproc, t_n_sys, rcvbufR_d, rcvbufL_d, sndbuf_d,
        x_cur_d);

  } while (++iter < max_iter && need_iter);

  cudaMemcpy(sndbuf_d + t_n_sys * 2 * 4, x_cur_d, t_n_sys * 2 * sizeof(Float),
             cudaMemcpyDeviceToDevice);

  rcvbufR_h = rcvbuf_h + t_n_sys * nvar_per_sys;
  rcvbufR_d = rcvbuf_d + t_n_sys * nvar_per_sys;
  // send solution to rank - 1 (left/UP)
  send_rows_to_nodes<communication_dir_t::UP, Float>(
      params, t_solvedim, 1, t_n_sys * 2, x_cur_d, sndbuf_h, nullptr, nullptr,
      rcvbufR_d, rcvbufR_h);
}

// ----------------------------------------------------------------------------
// Solver function
// ----------------------------------------------------------------------------

template <typename Float>
void pentadsolver_gpsv_batch(pentadsolver_handle_t params, const Float *ds,
                             const Float *dl, const Float *d, const Float *du,
                             const Float *dw, Float *x, const int *t_dims,
                             size_t t_ndims, int t_solvedim, void *t_buffer_h,
                             void *t_buffer_d) {

  size_t buff_size =
      std::accumulate(t_dims, t_dims + t_ndims, 1, std::multiplies<>());
  size_t n_sys = buff_size / t_dims[t_solvedim];
  Float *dss   = reinterpret_cast<Float *>(t_buffer_d) + buff_size * 0;
  Float *dll   = reinterpret_cast<Float *>(t_buffer_d) + buff_size * 1;
  Float *duu   = reinterpret_cast<Float *>(t_buffer_d) + buff_size * 2;
  Float *dww   = reinterpret_cast<Float *>(t_buffer_d) + buff_size * 3;
  constexpr int reduced_size_elem = 10; // 2 row per node 5 value per row
  Float *sndbuf_d                 = dww + buff_size;
  Float *rcvbuf_d                 = sndbuf_d + n_sys * reduced_size_elem;
  auto *sndbuf_h                  = reinterpret_cast<Float *>(t_buffer_h);
  Float *rcvbuf_h                 = sndbuf_h + n_sys * reduced_size_elem;
  gpsv_batched_forward(ds, dl, d, du, dw, x, dss, dll, duu, dww, sndbuf_d,
                       rcvbuf_d, t_dims, t_ndims, n_sys, t_solvedim);
  solve_reduced_pcr(params, rcvbuf_d, sndbuf_d, rcvbuf_h, sndbuf_h, t_solvedim,
                    n_sys);
  gpsv_batched_backward(dss, dll, duu, dww, x, sndbuf_d + 2 * 4 * n_sys,
                        rcvbuf_d + reduced_size_elem * n_sys, t_dims, t_ndims,
                        n_sys, t_solvedim);
}

// ----------------------------------------------------------------------------
// Pentadsolver context functions
// ----------------------------------------------------------------------------

void pentadsolver_create(pentadsolver_handle_t *handle, void *communicator,
                         int ndims, const int *num_procs) {

  // NOLINTNEXTLINE
  *handle = new pentadsolver_handle_implementation(
      *reinterpret_cast<MPI_Comm *>(communicator), ndims, num_procs);
}
void pentadsolver_destroy(pentadsolver_handle_t *handle) {
  // NOLINTNEXTLINE
  delete *handle;
}

// ----------------------------------------------------------------------------
// Buffer size calculation
// ----------------------------------------------------------------------------

template <typename Float>
void pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t /*handle*/, const Float * /*ds*/,
    const Float * /*dl*/, const Float * /*d*/, const Float * /*du*/,
    const Float * /*dw*/, const Float * /*x*/, const int *t_dims, size_t t_ndim,
    int t_solvedim, size_t *t_workspace_in_bytes_host,
    size_t *t_workspace_in_bytes_device) {

  size_t buff_size =
      std::accumulate(t_dims, t_dims + t_ndim, 1, std::multiplies<>());
  size_t n_sys                       = buff_size / t_dims[t_solvedim];
  constexpr int num_comm_bufs        = 3;
  constexpr int nvar_per_sys         = 10;
  constexpr int num_intermediate_buf = 4;

  // TODO fix sizes to the correct amounts
  *t_workspace_in_bytes_device = (n_sys * nvar_per_sys * num_comm_bufs +
                                  buff_size * num_intermediate_buf) *
                                 sizeof(Float);
  *t_workspace_in_bytes_host =
      (n_sys * nvar_per_sys * num_comm_bufs) * sizeof(Float);
}

// ----------------------------------------------------------------------------
// Adapter function implementations
// ----------------------------------------------------------------------------

void pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const double *ds, const double *dl,
    const double *d, const double *du, const double *dw, const double *x,
    const int *t_dims, int t_ndim, int t_solvedim,
    size_t *t_workspace_in_bytes_host, size_t *t_workspace_in_bytes_device) {
  pentadsolver_gpsv_batch_buffer_size_ext(
      handle, ds, dl, d, du, dw, x, t_dims, static_cast<size_t>(t_ndim),
      t_solvedim, t_workspace_in_bytes_host, t_workspace_in_bytes_device);
}

void pentadsolver_D_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const double *ds, const double *dl,
    const double *d, const double *du, const double *dw, double *x,
    const int *t_dims, int t_ndim, int t_solvedim,
    size_t *t_workspace_in_bytes_host, size_t *t_workspace_in_bytes_device) {
  pentadsolver_gpsv_batch_buffer_size_ext(
      handle, ds, dl, d, du, dw, x, t_dims, t_ndim, t_solvedim,
      t_workspace_in_bytes_host, t_workspace_in_bytes_device);
}

void pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const float *ds, const float *dl,
    const float *d, const float *du, const float *dw, const float *x,
    const int *t_dims, int t_ndim, int t_solvedim,
    size_t *t_workspace_in_bytes_host, size_t *t_workspace_in_bytes_device) {
  pentadsolver_gpsv_batch_buffer_size_ext(
      handle, ds, dl, d, du, dw, x, t_dims, static_cast<size_t>(t_ndim),
      t_solvedim, t_workspace_in_bytes_host, t_workspace_in_bytes_device);
}

void pentadsolver_S_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const float *ds, const float *dl,
    const float *d, const float *du, const float *dw, float *x,
    const int *t_dims, int t_ndim, int t_solvedim,
    size_t *t_workspace_in_bytes_host, size_t *t_workspace_in_bytes_device) {
  pentadsolver_gpsv_batch_buffer_size_ext(
      handle, ds, dl, d, du, dw, x, t_dims, t_ndim, t_solvedim,
      t_workspace_in_bytes_host, t_workspace_in_bytes_device);
}

void pentadsolver_gpsv_batch(pentadsolver_handle_t handle, const double *ds,
                             const double *dl, const double *d,
                             const double *du, const double *dw, double *x,
                             const int *t_dims, int t_ndim, int t_solvedim,
                             void *t_buffer_h, void *t_buffer_d) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims,
                          static_cast<size_t>(t_ndim), t_solvedim, t_buffer_h,
                          t_buffer_d);
}

void pentadsolver_D_gpsv_batch(pentadsolver_handle_t handle, const double *ds,
                               const double *dl, const double *d,
                               const double *du, const double *dw, double *x,
                               const int *t_dims, int t_ndim, int t_solvedim,
                               void *t_buffer_h, void *t_buffer_d) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims, t_ndim,
                          t_solvedim, t_buffer_h, t_buffer_d);
}

void pentadsolver_gpsv_batch(pentadsolver_handle_t handle, const float *ds,
                             const float *dl, const float *d, const float *du,
                             const float *dw, float *x, const int *t_dims,
                             int t_ndim, int t_solvedim, void *t_buffer_h,
                             void *t_buffer_d) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims,
                          static_cast<size_t>(t_ndim), t_solvedim, t_buffer_h,
                          t_buffer_d);
}

void pentadsolver_S_gpsv_batch(pentadsolver_handle_t handle, const float *ds,
                               const float *dl, const float *d, const float *du,
                               const float *dw, float *x, const int *t_dims,
                               int t_ndim, int t_solvedim, void *t_buffer_h,
                               void *t_buffer_d) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims, t_ndim,
                          t_solvedim, t_buffer_h, t_buffer_d);
}
