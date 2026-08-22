#include <mpi.h>                   // for MPI_Request, MPI_REQUEST_NULL
#include <cstddef>                 // for size_t
#include <array>                   // for array
#include <cassert>                 // for assert
#include <cmath>                   // for ceil, log2
#include <functional>              // for multiplies
#include <numeric>                 // for accumulate
#include <type_traits>             // for is_same
#include <vector>                  // for vector
#include "pentadsolver.hpp"        // for pentadsolver_gpsv_batch, pentadso...
#include "pentadsolver_handle.hpp" // for pentadsolver_handle_implementation

namespace {
template <typename REAL>
const MPI_Datatype mpi_datatype =
    std::is_same<REAL, double>::value ? MPI_DOUBLE : MPI_FLOAT;
constexpr size_t full_line_l = 6;
enum class communication_dir_t { DOWN = 1, UP = 2, ALL = 3 };
} // namespace

template <typename Float>
void shift_uw(const Float *ds, const Float *dl, const Float *du,
              const Float *dw, const Float *xx, size_t r1_idx,
              std::array<Float, full_line_l> &r0) {
  // index namin olds for r1, in case of r0: l, d, tmp, u, w x
  constexpr size_t s = 0;
  constexpr size_t l = 1;
  constexpr size_t d = 2;
  constexpr size_t u = 3;
  constexpr size_t w = 4;
  constexpr size_t x = 5;
  Float u0_tmp       = r0[u];
  r0[x]              = r0[x] - u0_tmp * xx[r1_idx];
  r0[l]              = r0[l] - u0_tmp * ds[r1_idx];
  r0[d]              = r0[d] - u0_tmp * dl[r1_idx];
  r0[u]              = r0[w] - u0_tmp * du[r1_idx];
  r0[w]              = 00 - u0_tmp * dw[r1_idx];
}

template <typename Float>
void pack_first_rows_forward(Float ds0, std::array<Float, full_line_l> &r0,
                             std::array<Float, full_line_l> &r1,
                             Float *sndbuf) {
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
  //  note: we add ds to the beginning of the array so indexing will change
  Float u0_tmp         = r0[d];
  r0[d]                = r0[l] - u0_tmp * r1[l]; // ddi
  sndbuf[2 * s]        = ds0 / r0[d];
  sndbuf[2 * l]        = (r0[s] - u0_tmp * r1[s]) / r0[d];
  sndbuf[2 * usnd]     = (r0[u] - u0_tmp * r1[u]) / r0[d];
  sndbuf[2 * wsnd]     = (r0[w] - u0_tmp * r1[w]) / r0[d];
  sndbuf[2 * xsnd]     = (r0[x] - u0_tmp * r1[x]) / r0[d];
  sndbuf[2 * s + 1]    = r1[s];
  sndbuf[2 * l + 1]    = r1[l];
  sndbuf[2 * usnd + 1] = r1[u];
  sndbuf[2 * wsnd + 1] = r1[w];
  sndbuf[2 * xsnd + 1] = r1[x];
}

template <typename Float>
void shift_sl_strided(const Float *ds, const Float *dl, const Float *d,
                      const Float *du, const Float *dw, Float *xx, Float *dss,
                      Float *dll, Float *duu, Float *dww, size_t idx,
                      size_t t_stride) {
  size_t ri   = idx * t_stride;
  size_t rim1 = (idx - 1) * t_stride;
  size_t rim2 = (idx - 2) * t_stride;
  if (idx <= 3) {
    Float si = ds[ri];
    Float li = dl[ri];
    Float di = d[ri] - li * duu[rim1];
    xx[ri]   = (xx[ri] - li * xx[rim1]) / di;
    dss[ri]  = (-li * dss[rim1]) / di;
    dll[ri]  = (si - li * dll[rim1]) / di;
    duu[ri]  = (du[ri] - li * dww[rim1]) / di;
    dww[ri]  = (dw[ri]) / di;
  } else {
    Float si = ds[ri];
    Float li = dl[ri] - si * duu[rim2];
    Float di = d[ri] - si * dww[rim2] - li * duu[rim1];
    xx[ri]   = (xx[ri] - si * xx[rim2] - li * xx[rim1]) / di;
    dss[ri]  = (-si * dss[rim2] - li * dss[rim1]) / di;
    dll[ri]  = (-si * dll[rim2] - li * dll[rim1]) / di;
    duu[ri]  = (du[ri] - li * dww[rim1]) / di;
    dww[ri]  = (dw[ri]) / di;
  }
}

template <typename Float>
void gpsv_forward_x(const Float *ds, const Float *dl, const Float *d,
                    const Float *du, const Float *dw, Float *x, Float *dss,
                    Float *dll, Float *duu, Float *dww, Float *sndbuf,
                    Float *snd_bottom_buf, size_t t_sys_size) {
  assert(t_sys_size >= 4); // NOLINT
  // row 0:
  // M[0, 1] cannot be shifted until the end
  // So we add a temp value to the line and shift it like row 1 until the end
  // Finally we will shift du as well and add back ds[0]
  std::array<Float, full_line_l> r0 = {dl[0], d[0], du[0], dw[0], 0, x[0]};
  // row 1
  std::array<Float, full_line_l> r1 = {ds[1], dl[1], d[1], du[1], dw[1], x[1]};
  // row 2: norm, shift u, w in row 0, 1
  x[2]   = x[2] / d[2];
  dss[2] = ds[2] / d[2];
  dll[2] = dl[2] / d[2];
  duu[2] = du[2] / d[2];
  dww[2] = dw[2] / d[2];
  shift_uw(dss, dll, duu, dww, x, 2, r0);
  shift_uw(dss, dll, duu, dww, x, 2, r1);
  // row 3
  Float si  = ds[3];
  Float li  = dl[3];
  Float ddi = d[3] - li * duu[2];
  x[3]      = (x[3] - li * x[2]) / ddi;
  dss[3]    = (-li * dss[2]) / ddi;
  dll[3]    = (si - li * dll[2]) / ddi;
  duu[3]    = (du[3] - li * dww[2]) / ddi;
  dww[3]    = (dw[3]) / ddi;
  shift_uw(dss, dll, duu, dww, x, 3, r0);
  shift_uw(dss, dll, duu, dww, x, 3, r1);
  // from row 4 till t_sys_size:
  // shift {s, l} with row i-2 and i-1, norm, shift u0, w0
  for (int i = 4; i < t_sys_size; ++i) {
    si     = ds[i];
    li     = dl[i] - si * duu[i - 2];
    ddi    = d[i] - si * dww[i - 2] - li * duu[i - 1];
    x[i]   = (x[i] - si * x[i - 2] - li * x[i - 1]) / ddi;
    dss[i] = (-si * dss[i - 2] - li * dss[i - 1]) / ddi;
    dll[i] = (-si * dll[i - 2] - li * dll[i - 1]) / ddi;
    duu[i] = (du[i] - li * dww[i - 1]) / ddi;
    dww[i] = (dw[i]) / ddi;
    // shift u0, w0
    shift_uw(dss, dll, duu, dww, x, i, r0);
    shift_uw(dss, dll, duu, dww, x, i, r1);
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
  size_t i = t_sys_size - 2;
  li       = duu[i];
  x[i]     = x[i] - li * x[i + 1];
  dss[i]   = dss[i] - li * dss[i + 1];
  dll[i]   = dll[i] - li * dll[i + 1];
  duu[i]   = dww[i] - li * duu[i + 1];
  dww[i]   = -li * dww[i + 1];
  // copy last two rows to combuf for reduced step
  snd_bottom_buf[0] = dss[i];
  snd_bottom_buf[1] = dss[i + 1];
  snd_bottom_buf[2] = dll[i];
  snd_bottom_buf[3] = dll[i + 1];
  snd_bottom_buf[4] = duu[i];
  snd_bottom_buf[5] = duu[i + 1]; // NOLINT
  snd_bottom_buf[6] = dww[i];     // NOLINT
  snd_bottom_buf[7] = dww[i + 1]; // NOLINT
  snd_bottom_buf[8] = x[i];       // NOLINT
  snd_bottom_buf[9] = x[i + 1];   // NOLINT

  // Prepare layout of the first two row for comm
  pack_first_rows_forward(ds[0], r0, r1, sndbuf);
}

template <typename Float>
void gpsv_forward_strided(const Float *ds, const Float *dl, const Float *d,
                          const Float *du, const Float *dw, Float *x,
                          Float *dss, Float *dll, Float *duu, Float *dww,
                          Float *sndbuf, Float *snd_bottom_buf,
                          size_t t_sys_size, size_t t_stride) {
  assert(t_sys_size >= 4); // NOLINT
  // row 0:
  // M[0, 1] cannot be shifted until the end
  // So we add a temp value to the line and shift it like row 1 until the end
  // Finally we will shift du as well and add back ds[0]
  std::array<Float, full_line_l> r0 = {dl[0], d[0], du[0], dw[0], 0, x[0]};
  // row 1
  std::array<Float, full_line_l> r1 = {ds[t_stride], dl[t_stride], d[t_stride],
                                       du[t_stride], dw[t_stride], x[t_stride]};
  // row 2: norm, shift u, w in row 0, 1
  x[2 * t_stride]   = x[2 * t_stride] / d[2 * t_stride];
  dss[2 * t_stride] = ds[2 * t_stride] / d[2 * t_stride];
  dll[2 * t_stride] = dl[2 * t_stride] / d[2 * t_stride];
  duu[2 * t_stride] = du[2 * t_stride] / d[2 * t_stride];
  dww[2 * t_stride] = dw[2 * t_stride] / d[2 * t_stride];
  shift_uw(dss, dll, duu, dww, x, 2 * t_stride, r0);
  shift_uw(dss, dll, duu, dww, x, 2 * t_stride, r1);
  // row 3
  shift_sl_strided(ds, dl, d, du, dw, x, dss, dll, duu, dww, 3, t_stride);
  shift_uw(dss, dll, duu, dww, x, 3 * t_stride, r0);
  shift_uw(dss, dll, duu, dww, x, 3 * t_stride, r1);
  // from row 4 till t_sys_size:
  // shift {s, l} with row i-2 and i-1, norm, shift u0, w0
  for (int i = 4; i < t_sys_size; ++i) {
    shift_sl_strided(ds, dl, d, du, dw, x, dss, dll, duu, dww, i, t_stride);
    // shift u0, w0
    shift_uw(dss, dll, duu, dww, x, i * t_stride, r0);
    shift_uw(dss, dll, duu, dww, x, i * t_stride, r1);
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
  size_t i          = t_sys_size - 2;
  Float li          = duu[i * t_stride];
  x[i * t_stride]   = x[i * t_stride] - li * x[(i + 1) * t_stride];
  dss[i * t_stride] = dss[i * t_stride] - li * dss[(i + 1) * t_stride];
  dll[i * t_stride] = dll[i * t_stride] - li * dll[(i + 1) * t_stride];
  duu[i * t_stride] = dww[i * t_stride] - li * duu[(i + 1) * t_stride];
  dww[i * t_stride] = -li * dww[(i + 1) * t_stride];
  // copy last two rows to combuf for reduced step
  snd_bottom_buf[0] = dss[i * t_stride];
  snd_bottom_buf[1] = dss[(i + 1) * t_stride];
  snd_bottom_buf[2] = dll[i * t_stride];
  snd_bottom_buf[3] = dll[(i + 1) * t_stride];
  snd_bottom_buf[4] = duu[i * t_stride];
  snd_bottom_buf[5] = duu[(i + 1) * t_stride]; // NOLINT
  snd_bottom_buf[6] = dww[i * t_stride];       // NOLINT
  snd_bottom_buf[7] = dww[(i + 1) * t_stride]; // NOLINT
  snd_bottom_buf[8] = x[i * t_stride];         // NOLINT
  snd_bottom_buf[9] = x[(i + 1) * t_stride];   // NOLINT

  // Prepare layout of the first two row for comm
  pack_first_rows_forward(ds[0], r0, r1, sndbuf);
}

template <typename Float>
void gpsv_forward_batched_x(const Float *ds, const Float *dl, const Float *d,
                            const Float *du, const Float *dw, Float *x,
                            Float *dss, Float *dll, Float *duu, Float *dww,
                            Float *comm_buf, Float *snd_bottom_buf,
                            size_t t_n_sys, int t_sys_size) {
#pragma omp parallel for
  for (int sys_id = 0; sys_id < t_n_sys; ++sys_id) {
    int sys_start              = sys_id * t_sys_size;
    constexpr int nvar_per_sys = 10;
    int sys_comm_start         = sys_id * nvar_per_sys;
    gpsv_forward_x(ds + sys_start, dl + sys_start, d + sys_start,
                   du + sys_start, dw + sys_start, x + sys_start,
                   dss + sys_start, dll + sys_start, duu + sys_start,
                   dww + sys_start, comm_buf + sys_comm_start,
                   snd_bottom_buf + sys_comm_start, t_sys_size);
  }
}

template <typename Float>
void gpsv_forward_batched_y(const Float *ds, const Float *dl, const Float *d,
                            const Float *du, const Float *dw, Float *x,
                            Float *dss, Float *dll, Float *duu, Float *dww,
                            Float *comm_buf, Float *snd_bottom_buf,
                            int t_n_sys_in, int t_sys_size, int t_n_sys_out) {
#pragma omp parallel for
  for (int i = 0; i < t_n_sys_out; ++i) {
    for (int j = 0; j < t_n_sys_in; ++j) {
      size_t sys_start           = i * t_n_sys_in * t_sys_size + j;
      constexpr int nvar_per_sys = 10;
      int sys_comm_start         = (i * t_n_sys_in + j) * nvar_per_sys;
      gpsv_forward_strided(
          ds + sys_start, dl + sys_start, d + sys_start, du + sys_start,
          dw + sys_start, x + sys_start, dss + sys_start, dll + sys_start,
          duu + sys_start, dww + sys_start, comm_buf + sys_comm_start,
          snd_bottom_buf + sys_comm_start, t_sys_size, t_n_sys_in);
    }
  }
}

template <typename Float>
void gpsv_forward_batched_outermost(const Float *ds, const Float *dl,
                                    const Float *d, const Float *du,
                                    const Float *dw, Float *x, Float *dss,
                                    Float *dll, Float *duu, Float *dww,
                                    Float *comm_buf, Float *snd_bottom_buf,
                                    size_t t_n_sys, int t_sys_size) {
#pragma omp parallel for
  for (int sys_id = 0; sys_id < t_n_sys; ++sys_id) {
    int sys_start              = sys_id;
    constexpr int nvar_per_sys = 10;
    int sys_comm_start         = sys_id * nvar_per_sys;
    gpsv_forward_strided(ds + sys_start, dl + sys_start, d + sys_start,
                         du + sys_start, dw + sys_start, x + sys_start,
                         dss + sys_start, dll + sys_start, duu + sys_start,
                         dww + sys_start, comm_buf + sys_comm_start,
                         snd_bottom_buf + sys_comm_start, t_sys_size, t_n_sys);
  }
}

template <typename Float>
void gpsv_backward_x(const Float *dss, const Float *dll, const Float *duu,
                     const Float *dww, Float x0, Float x1, Float xp1, Float xp2,
                     Float *x, int t_sys_size) {
  x[0] = x0;
  x[1] = x1;
  // last two rows:
  {
    int i = t_sys_size - 1;
    x[i]  = x[i] - dww[i] * xp2 - duu[i] * xp1 - dll[i] * x1 - dss[i] * x0;
    i     = t_sys_size - 2;
    x[i]  = x[i] - dww[i] * xp2 - duu[i] * xp1 - dll[i] * x1 - dss[i] * x0;
    xp1   = x[i];
    xp2   = x[i + 1];
  }

  for (int i = t_sys_size - 3; i > 1; --i) {
    x[i] = x[i] - dww[i] * xp2 - duu[i] * xp1 - dll[i] * x1 - dss[i] * x0;
    xp2  = xp1;
    xp1  = x[i];
  }
}

template <typename Float>
void gpsv_backward_strided(const Float *dss, const Float *dll, const Float *duu,
                           const Float *dww, Float x0, Float x1, Float xp1,
                           Float xp2, Float *x, int t_sys_size,
                           size_t t_stride) {
  x[0]        = x0;
  x[t_stride] = x1;
  // last two rows:
  {
    size_t i = (t_sys_size - 1) * t_stride;
    x[i]     = x[i] - dww[i] * xp2 - duu[i] * xp1 - dll[i] * x1 - dss[i] * x0;
    i        = (t_sys_size - 2) * t_stride;
    x[i]     = x[i] - dww[i] * xp2 - duu[i] * xp1 - dll[i] * x1 - dss[i] * x0;
    xp1      = x[i];
    xp2      = x[i + t_stride];
  }

  for (int i = t_sys_size - 3; i > 1; --i) {
    size_t idx = i * t_stride;
    x[idx]     = x[idx] - dww[idx] * xp2 - duu[idx] * xp1 - dll[idx] * x1 -
             dss[idx] * x0;
    xp2 = xp1;
    xp1 = x[idx];
  }
}

template <typename Float>
void gpsv_backward_batched_x(const Float *dss, const Float *dll,
                             const Float *duu, const Float *dww, Float *x,
                             Float *comm_buf, Float *rcvbufR, size_t t_n_sys,
                             int t_sys_size) {
#pragma omp parallel for
  for (int sys_id = 0; sys_id < t_n_sys; ++sys_id) {
    int sys_start              = sys_id * t_sys_size;
    constexpr int nvar_per_sys = 10;
    int sys_comm_x_start       = (sys_id + 1) * nvar_per_sys - 2;
    gpsv_backward_x(dss + sys_start, dll + sys_start, duu + sys_start,
                    dww + sys_start, comm_buf[sys_comm_x_start],
                    comm_buf[sys_comm_x_start + 1], rcvbufR[2 * sys_id + 0],
                    rcvbufR[2 * sys_id + 1], x + sys_start, t_sys_size);
  }
}

template <typename Float>
void gpsv_backward_batched_outermost(const Float *dss, const Float *dll,
                                     const Float *duu, const Float *dww,
                                     Float *x, Float *comm_buf, Float *rcvbufR,
                                     size_t t_n_sys, int t_sys_size) {
#pragma omp parallel for
  for (int i = 0; i < t_n_sys; ++i) {
    size_t sys_start           = i;
    constexpr int nvar_per_sys = 10;
    int sys_comm_x_start       = (i + 1) * nvar_per_sys - 2;
    gpsv_backward_strided(dss + sys_start, dll + sys_start, duu + sys_start,
                          dww + sys_start, comm_buf[sys_comm_x_start],
                          comm_buf[sys_comm_x_start + 1], rcvbufR[2 * i + 0],
                          rcvbufR[2 * i + 1], x + sys_start, t_sys_size,
                          t_n_sys);
  }
}

template <typename Float>
void gpsv_backward_batched_middle(const Float *dss, const Float *dll,
                                  const Float *duu, const Float *dww, Float *x,
                                  Float *comm_buf, Float *rcvbufR,
                                  size_t t_n_sys_in, size_t t_sys_size,
                                  size_t t_n_sys_out) {

#pragma omp parallel for collapse(2)
  for (int i = 0; i < t_n_sys_out; ++i) {
    for (int j = 0; j < t_n_sys_in; ++j) {
      size_t sys_start           = i * t_n_sys_in * t_sys_size + j;
      constexpr int nvar_per_sys = 10;
      size_t sys_id              = i * t_n_sys_in + j;
      size_t sys_comm_x_start    = (sys_id + 1) * nvar_per_sys - 2;
      gpsv_backward_strided(dss + sys_start, dll + sys_start, duu + sys_start,
                            dww + sys_start, comm_buf[sys_comm_x_start],
                            comm_buf[sys_comm_x_start + 1],
                            rcvbufR[2 * sys_id + 0], rcvbufR[2 * sys_id + 1],
                            x + sys_start, t_sys_size, t_n_sys_in);
    }
  }
}

template <typename Float>
void shift_sl_reduced(int rank, const Float *rcvbuf, Float *top_buf) {
  constexpr size_t s = 0;
  constexpr size_t l = 2;
  constexpr size_t u = 4;
  constexpr size_t w = 6;
  constexpr size_t x = 8;
  Float r1l_tmp      = top_buf[l + 1];
  top_buf[l + 1]     = 0.0;
  if (rank > 0) {
    // row 1
    Float si       = top_buf[s + 1];
    Float di_tmp   = 1 - si * rcvbuf[w + 1];
    r1l_tmp        = (r1l_tmp - si * rcvbuf[u + 1]) / di_tmp;
    top_buf[s + 1] = (-si * rcvbuf[s + 1]) / di_tmp;
    top_buf[l + 1] = (-si * rcvbuf[l + 1]) / di_tmp;
    top_buf[u + 1] = (top_buf[u + 1]) / di_tmp;
    top_buf[w + 1] = (top_buf[w + 1]) / di_tmp;
    top_buf[x + 1] = (top_buf[x + 1] - si * rcvbuf[x + 1]) / di_tmp;
    // row 0
    si            = top_buf[s];
    Float li      = top_buf[l];
    Float r0u_tmp = -si * rcvbuf[w] - li * rcvbuf[w + 1];
    di_tmp        = 1 - si * rcvbuf[u] - li * rcvbuf[u + 1] - r0u_tmp * r1l_tmp;
    top_buf[s] =
        (-si * rcvbuf[s] - li * rcvbuf[s + 1] - r0u_tmp * top_buf[s + 1]) /
        di_tmp;
    top_buf[l] =
        (-si * rcvbuf[l] - li * rcvbuf[l + 1] - r0u_tmp * top_buf[l + 1]) /
        di_tmp;
    top_buf[u] = (top_buf[u] - r0u_tmp * top_buf[u + 1]) / di_tmp;
    top_buf[w] = (top_buf[w] - r0u_tmp * top_buf[w + 1]) / di_tmp;
    top_buf[x] = (top_buf[x] - si * rcvbuf[x] - li * rcvbuf[x + 1] -
                  r0u_tmp * top_buf[x + 1]) /
                 di_tmp;
  } else {
    // align indexing of s, l in row 1 to row 0
    top_buf[l + 1] = top_buf[s + 1];
    top_buf[s + 1] = 0.0;
  }
  top_buf[s + 1] = top_buf[s + 1] - r1l_tmp * top_buf[s];
  top_buf[l + 1] = top_buf[l + 1] - r1l_tmp * top_buf[l];
  top_buf[u + 1] = top_buf[u + 1] - r1l_tmp * top_buf[u];
  top_buf[w + 1] = top_buf[w + 1] - r1l_tmp * top_buf[w];
  top_buf[x + 1] = top_buf[x + 1] - r1l_tmp * top_buf[x];
}

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

template <typename Float>
void eliminate_bottom_rows_from_reduced(pentadsolver_handle_t params,
                                        int t_solvedim, int t_n_sys,
                                        const Float *snd_bottom_buf,
                                        Float *rcvbuf, Float *top_buf) {
  int rank                   = params->mpi_coords[t_solvedim];
  constexpr int nvar_per_sys = 10;
  send_rows_to_nodes<communication_dir_t::DOWN, Float>(
      params, t_solvedim, 1, t_n_sys * nvar_per_sys, snd_bottom_buf, rcvbuf,
      nullptr);
// s l 1 0 u w         -> s l 1 0 u w
// s l 0 1 u w         -> s l 0 1 u w
// ------------------- -> ---------------
//     s l 1 0 u w     -> s l     1 0 u w
//       s l 1 u w     -> s l     0 1 u w
// ------------------- -> ---------------
#pragma omp parallel for
  for (int i = 0; i < t_n_sys; ++i) {
    shift_sl_reduced(rank, rcvbuf + i * nvar_per_sys,
                     top_buf + i * nvar_per_sys);
  }
}

// Solve reduced system with PCR algorithm.
// sndbuf holds the first two row of each system after the forward pass
// as an input and the solutions at the x positions as an output. sndbuf
// stores each system packed as: s0, s1, l0, l1, u0, u1, w0, w1, x0, x1 For
// each node for each system the first and last row of create the reduced
// system, d_i = 1 everywhere. rcvbuf must be an array with size at least 2 *
// 10 * n_sys. rcvbuf[0:nsys], rcvbuf[nsys+1:2*nsys] will hold the solution
// xm1, xp1 on exit, respectively
template <typename Float>
inline void solve_reduced_pcr(pentadsolver_handle_t params, Float *rcvbuf,
                              Float *sndbuf, int t_solvedim, int t_n_sys) {
  int rank                   = params->mpi_coords[t_solvedim];
  int nproc                  = params->num_mpi_procs[t_solvedim];
  constexpr int nvar_per_sys = 10;
  Float *rcvbufL             = rcvbuf;
  // eliminate_bottom_rows require whole systems in messages
  Float *rcvbufR = rcvbuf + t_n_sys * nvar_per_sys;

  // eliminate 2 rows from reduced system
  eliminate_bottom_rows_from_reduced(params, t_solvedim, t_n_sys, rcvbufL,
                                     rcvbufR, sndbuf);

  int P    = std::ceil(std::log2((double)nproc));
  int dist = 1;
  // perform pcr
  // loop for for comm
  for (int p = 0; p < P; ++p) {
    send_rows_to_nodes<communication_dir_t::ALL, Float>(
        params, t_solvedim, dist, t_n_sys * nvar_per_sys, sndbuf, rcvbufL,
        rcvbufR);
    int leftrank  = rank - dist;
    int rightrank = rank + dist;

// PCR algorithm
#pragma omp parallel for
    for (int id = 0; id < t_n_sys; id++) {
      // s l 1 0 u w         -> s l 1 0 u w
      // s l 0 1 u w         -> s l 0 1 u w
      // ------------------- -> -------------------
      //     s l 1 0 u w     -> s l     1 0     u w
      //     s l 0 1 u w     -> s l     0 1     u w
      // ------------------- -> -------------------
      //         s l 1 0 u w ->         s l 1 0 u w
      //         s l 0 1 u w ->         s l 0 1 u w
      int sidx = id * nvar_per_sys;

      constexpr size_t s                      = 0;
      constexpr size_t l                      = 1;
      constexpr size_t u                      = 2;
      constexpr size_t w                      = 3;
      constexpr size_t x                      = 4;
      std::array<Float, nvar_per_sys / 2> rm2 = {};
      std::array<Float, nvar_per_sys / 2> rm1 = {};
      std::array<Float, nvar_per_sys / 2> rp1 = {};
      std::array<Float, nvar_per_sys / 2> rp2 = {};
      if (leftrank >= 0) {
        for (int i = 0; i < nvar_per_sys / 2; ++i) {
          rm2[i] = rcvbufL[sidx + 2 * i + 0];
          rm1[i] = rcvbufL[sidx + 2 * i + 1];
        }
      }
      if (rightrank < nproc) {
        for (int i = 0; i < nvar_per_sys / 2; ++i) {
          rp1[i] = rcvbufR[sidx + 2 * i + 0];
          rp2[i] = rcvbufR[sidx + 2 * i + 1];
        }
      }
      std::array<Float, nvar_per_sys / 2> r0 = {
          sndbuf[sidx + 2 * s], sndbuf[sidx + 2 * l], sndbuf[sidx + 2 * u],
          sndbuf[sidx + 2 * w], sndbuf[sidx + 2 * x]};
      std::array<Float, nvar_per_sys / 2> r1 = {
          sndbuf[sidx + 2 * s + 1], sndbuf[sidx + 2 * l + 1],
          sndbuf[sidx + 2 * u + 1], sndbuf[sidx + 2 * w + 1],
          sndbuf[sidx + 2 * x + 1]};
      // shift by one:
      Float d0 =
          1 - r0[l] * rm1[u] - r0[s] * rm2[u] - r0[u] * rp1[s] - r0[w] * rp2[s];
      Float tmp0 =
          -r0[l] * rm1[w] - r0[s] * rm2[w] - r0[u] * rp1[l] - r0[w] * rp2[l];
      Float d1 =
          1 - r1[l] * rm1[w] - r1[s] * rm2[w] - r1[u] * rp1[l] - r1[w] * rp2[l];
      Float tmp1 =
          -r1[l] * rm1[u] - r1[s] * rm2[u] - r1[u] * rp1[s] - r1[w] * rp2[s];
      sndbuf[sidx + 2 * s + 0] = -r0[l] * rm1[s] - r0[s] * rm2[s];
      sndbuf[sidx + 2 * l + 0] = -r0[l] * rm1[l] - r0[s] * rm2[l];
      sndbuf[sidx + 2 * u + 0] = -r0[u] * rp1[u] - r0[w] * rp2[u];
      sndbuf[sidx + 2 * w + 0] = -r0[u] * rp1[w] - r0[w] * rp2[w];
      sndbuf[sidx + 2 * x + 0] = r0[x] - r0[u] * rp1[x] - r0[w] * rp2[x] -
                                 r0[l] * rm1[x] - r0[s] * rm2[x];
      sndbuf[sidx + 2 * s + 1] = -r1[l] * rm1[s] - r1[s] * rm2[s];
      sndbuf[sidx + 2 * l + 1] = -r1[l] * rm1[l] - r1[s] * rm2[l];
      sndbuf[sidx + 2 * u + 1] = -r1[u] * rp1[u] - r1[w] * rp2[u];
      sndbuf[sidx + 2 * w + 1] = -r1[u] * rp1[w] - r1[w] * rp2[w];
      sndbuf[sidx + 2 * x + 1] = r1[x] - r1[u] * rp1[x] - r1[w] * rp2[x] -
                                 r1[l] * rm1[x] - r1[s] * rm2[x];
      // zero out tmp values
      //     s l d T u w     -> s l 1 0 u w
      //     s l T d u w     -> s l T d u w
      Float coeff = tmp0 / d1;
      d0          = d0 - coeff * tmp1;
      sndbuf[sidx + 2 * s + 0] =
          (sndbuf[sidx + 2 * s + 0] - sndbuf[sidx + 2 * s + 1] * coeff) / d0;
      sndbuf[sidx + 2 * l + 0] =
          (sndbuf[sidx + 2 * l + 0] - sndbuf[sidx + 2 * l + 1] * coeff) / d0;
      sndbuf[sidx + 2 * u + 0] =
          (sndbuf[sidx + 2 * u + 0] - sndbuf[sidx + 2 * u + 1] * coeff) / d0;
      sndbuf[sidx + 2 * w + 0] =
          (sndbuf[sidx + 2 * w + 0] - sndbuf[sidx + 2 * w + 1] * coeff) / d0;
      sndbuf[sidx + 2 * x + 0] =
          (sndbuf[sidx + 2 * x + 0] - sndbuf[sidx + 2 * x + 1] * coeff) / d0;
      //     s l 1 0 u w     -> s l 1 0 u w
      //     s l T d u w     -> s l 0 1 u w
      sndbuf[sidx + 2 * s + 1] =
          (sndbuf[sidx + 2 * s + 1] - sndbuf[sidx + 2 * s + 0] * tmp1) / d1;
      sndbuf[sidx + 2 * l + 1] =
          (sndbuf[sidx + 2 * l + 1] - sndbuf[sidx + 2 * l + 0] * tmp1) / d1;
      sndbuf[sidx + 2 * u + 1] =
          (sndbuf[sidx + 2 * u + 1] - sndbuf[sidx + 2 * u + 0] * tmp1) / d1;
      sndbuf[sidx + 2 * w + 1] =
          (sndbuf[sidx + 2 * w + 1] - sndbuf[sidx + 2 * w + 0] * tmp1) / d1;
      sndbuf[sidx + 2 * x + 1] =
          (sndbuf[sidx + 2 * x + 1] - sndbuf[sidx + 2 * x + 0] * tmp1) / d1;
    }

    // done
    dist = dist << 1; // NOLINT
  }
  // pack new sndbuf:
#pragma omp parallel for
  for (int i = 0; i < t_n_sys; ++i) {
    constexpr int x0_idx = 8;
    constexpr int x1_idx = 9;
    rcvbufL[2 * i + 0]   = sndbuf[i * nvar_per_sys + x0_idx];
    rcvbufL[2 * i + 1]   = sndbuf[i * nvar_per_sys + x1_idx];
    rcvbufR[2 * i + 0]   = 0.0;
    rcvbufR[2 * i + 1]   = 0.0;
  }
  // send solution to rank - 1 (left/UP)
  send_rows_to_nodes<communication_dir_t::UP, Float>(
      params, t_solvedim, 1, t_n_sys * 2, rcvbufL, nullptr, rcvbufR);
}

template <typename Float>
inline void solve_reduced_jacobi(pentadsolver_handle_t params, Float *rcvbuf,
                                 Float *sndbuf, int t_solvedim, int t_n_sys) {
  int rank                   = params->mpi_coords[t_solvedim];
  int nproc                  = params->num_mpi_procs[t_solvedim];
  constexpr int nvar_per_sys = 10;
  constexpr int s_idx        = 0;
  constexpr int l_idx        = 2;
  constexpr int u_idx        = 4;
  constexpr int w_idx        = 6;
  constexpr int x_idx        = 8;
  Float *rcvbufL             = rcvbuf;
  Float *rcvbufR             = rcvbuf + t_n_sys * nvar_per_sys;

  // eliminate 2 rows from reduced system
  eliminate_bottom_rows_from_reduced(params, t_solvedim, t_n_sys, rcvbufL,
                                     rcvbufR, sndbuf);
  Float *x_cur  = rcvbuf;
  Float *x_prev = x_cur + 2 * t_n_sys;
  rcvbufL       = x_prev + 2 * t_n_sys;
  rcvbufR       = rcvbufL + 2 * t_n_sys;

  // TODO remove rank 0 from
#pragma omp parallel for
  for (size_t id = 0; id < t_n_sys; ++id) {
    x_cur[2 * id]     = sndbuf[id * nvar_per_sys + x_idx];
    x_cur[2 * id + 1] = sndbuf[id * nvar_per_sys + x_idx + 1];
  }

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
        params, t_solvedim, 1, t_n_sys * 2, x_cur, rcvbufL, rcvbufR);

#pragma omp parallel for
    for (size_t id = 0; id < t_n_sys; ++id) {
      // x_0 = (d_0 - s_{0} * x_{-2} - l_{0} * x_{-1} - u_{0} * x_{1} - w_{0} *
      // x_{2}) / b_0
      size_t sys_idx     = id * nvar_per_sys;
      x_prev[2 * id]     = x_cur[2 * id];
      x_prev[2 * id + 1] = x_cur[2 * id + 1];
      x_cur[2 * id]      = sndbuf[sys_idx + x_idx];
      x_cur[2 * id + 1]  = sndbuf[sys_idx + x_idx + 1];
      if (rank != nproc - 1) {
        Float dp1 = rcvbufR[2 * id];
        Float dp2 = rcvbufR[2 * id + 1];
        x_cur[2 * id] -=
            sndbuf[sys_idx + u_idx] * dp1 + sndbuf[sys_idx + w_idx] * dp2;
        x_cur[2 * id + 1] -= sndbuf[sys_idx + u_idx + 1] * dp1 +
                             sndbuf[sys_idx + w_idx + 1] * dp2;
      }
      if (rank) {
        Float dm2 = rcvbufL[2 * id];
        Float dm1 = rcvbufL[2 * id + 1];
        x_cur[2 * id] -=
            sndbuf[sys_idx + s_idx] * dm2 + sndbuf[sys_idx + l_idx] * dm1;
        x_cur[2 * id + 1] -= sndbuf[sys_idx + s_idx + 1] * dm2 +
                             sndbuf[sys_idx + l_idx + 1] * dm1;
      }
    }

    local_norm = 0;
#pragma omp parallel for reduction(max : local_norm)
    for (size_t i = 0; i < 2UL * t_n_sys; ++i) {
      double diff0 = x_prev[i] - x_cur[i];
      local_norm   = std::max(local_norm, diff0 * diff0);
    }
    if (iter > 0) { // skip until the first sum is ready
      MPI_Waitall(1, &norm_req, MPI_STATUS_IGNORE);
      norm_req         = MPI_REQUEST_NULL;
      global_norm_recv = sqrt(global_norm_recv / 2 * nproc);
      if (global_norm0 < 0) {
        global_norm0 = global_norm_recv;
      }
    }
    local_norm_send = local_norm;
    need_iter       = iter == 0 || (jacobi_atol < global_norm_recv &&
                              jacobi_rtol < global_norm_recv / global_norm0);
    if (iter + 1 < max_iter && need_iter) {
      MPI_Iallreduce(&local_norm_send, &global_norm_recv, 1, MPI_DOUBLE,
                     MPI_SUM, params->communicators[t_solvedim], &norm_req);
    }
  } while (++iter < max_iter && need_iter);

  // Change back rcvbufR (used in backward)
  rcvbufR = rcvbuf + t_n_sys * nvar_per_sys;

// pack results back:
#pragma omp parallel for
  for (size_t i = 0; i < t_n_sys; ++i) {
    sndbuf[i * nvar_per_sys + x_idx]     = x_cur[2 * i];
    sndbuf[i * nvar_per_sys + x_idx + 1] = x_cur[2 * i + 1];
    rcvbufR[2 * i + 0]                   = 0.0;
    rcvbufR[2 * i + 1]                   = 0.0;
  }
  // send solution to rank - 1 (left/UP)
  send_rows_to_nodes<communication_dir_t::UP, Float>(
      params, t_solvedim, 1, t_n_sys * 2, x_cur, nullptr, rcvbufR);
}

template <typename Float>
void gpsv_batched_forward(const Float *ds, const Float *dl, const Float *d,
                          const Float *du, const Float *dw, Float *x,
                          Float *dss, Float *dll, Float *duu, Float *dww,
                          Float *comm_buf, Float *snd_bottom_buf,
                          const int *t_dims, size_t t_ndims, size_t t_n_sys,
                          int t_solvedim) {
  if (t_solvedim == 0) {
    gpsv_forward_batched_x(ds, dl, d, du, dw, x, dss, dll, duu, dww, comm_buf,
                           snd_bottom_buf, t_n_sys, t_dims[t_solvedim]);
  } else if (t_solvedim == t_ndims - 1) {
    gpsv_forward_batched_outermost(ds, dl, d, du, dw, x, dss, dll, duu, dww,
                                   comm_buf, snd_bottom_buf, t_n_sys,
                                   t_dims[t_solvedim]);
  } else {
    size_t n_sys_in =
        std::accumulate(t_dims, t_dims + t_solvedim, 1, std::multiplies<>());
    size_t n_sys_out = std::accumulate(
        t_dims + t_solvedim + 1, t_dims + t_ndims, 1, std::multiplies<>());
    gpsv_forward_batched_y(ds, dl, d, du, dw, x, dss, dll, duu, dww, comm_buf,
                           snd_bottom_buf, n_sys_in, t_dims[t_solvedim],
                           n_sys_out);
  }
}

template <typename Float>
void gpsv_backward_batched(const Float *dss, const Float *dll, const Float *duu,
                           const Float *dww, Float *x, Float *comm_buf,
                           Float *rcvbufR, const int *t_dims, int t_ndims,
                           size_t t_n_sys, int t_solvedim) {
  if (t_solvedim == 0) {
    gpsv_backward_batched_x(dss, dll, duu, dww, x, comm_buf, rcvbufR, t_n_sys,
                            t_dims[t_solvedim]);
  } else if (t_solvedim == t_ndims - 1) {
    gpsv_backward_batched_outermost(dss, dll, duu, dww, x, comm_buf, rcvbufR,
                                    t_n_sys, t_dims[t_solvedim]);
  } else {
    size_t n_sys_in =
        std::accumulate(t_dims, t_dims + t_solvedim, 1, std::multiplies<>());
    size_t n_sys_out = std::accumulate(
        t_dims + t_solvedim + 1, t_dims + t_ndims, 1, std::multiplies<>());
    gpsv_backward_batched_middle(dss, dll, duu, dww, x, comm_buf, rcvbufR,
                                 n_sys_in, t_dims[t_solvedim], n_sys_out);
  }
}

template <typename Float>
void pentadsolver_gpsv_batch(pentadsolver_handle_t handle, const Float *ds,
                             const Float *dl, const Float *d, const Float *du,
                             const Float *dw, Float *x, const int *t_dims,
                             size_t t_ndims, int t_solvedim, void *t_buffer) {
  size_t buff_size =
      std::accumulate(t_dims, t_dims + t_ndims, 1, std::multiplies<>());
  size_t n_sys = buff_size / t_dims[t_solvedim];
  Float *dss   = reinterpret_cast<Float *>(t_buffer) + buff_size * 0;
  Float *dll   = reinterpret_cast<Float *>(t_buffer) + buff_size * 1;
  Float *duu   = reinterpret_cast<Float *>(t_buffer) + buff_size * 2;
  Float *dww   = reinterpret_cast<Float *>(t_buffer) + buff_size * 3;
  constexpr int reduced_size_elem = 10; // 2 row per node 5 value per row
  Float *sndbuf                   = dww + buff_size;
  Float *rcvbuf                   = sndbuf + n_sys * reduced_size_elem;
  gpsv_batched_forward(ds, dl, d, du, dw, x, dss, dll, duu, dww, sndbuf, rcvbuf,
                       t_dims, t_ndims, n_sys, t_solvedim);
  // solve_reduced_pcr(handle, rcvbuf, sndbuf, t_solvedim, n_sys);
  solve_reduced_jacobi(handle, rcvbuf, sndbuf, t_solvedim, n_sys);
  gpsv_backward_batched(dss, dll, duu, dww, x, sndbuf,
                        rcvbuf + reduced_size_elem * n_sys, t_dims, t_ndims,
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
[[nodiscard]] size_t pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t /*handle*/, const Float * /*ds*/,
    const Float * /*dl*/, const Float * /*d*/, const Float * /*du*/,
    const Float * /*dw*/, const Float * /*x*/, const int *t_dims,
    size_t t_ndims, int t_solvedim) {
  // Communication would need 3 buffers: 1 send 2 receive
  // Each buffer holds 2 rows for each system with d == 1 -> 5 elem per row
  // size = n_sys * 2 * 5 * 3
  // Intermediate results need 4 buffers: ds, dl, du, dw
  // each with the same size as the originals
  size_t buff_size =
      std::accumulate(t_dims, t_dims + t_ndims, 1, std::multiplies<>());
  size_t n_sys                       = buff_size / t_dims[t_solvedim];
  constexpr int num_comm_bufs        = 3;
  constexpr int nvar_per_sys         = 10;
  constexpr int num_intermediate_buf = 4;

  return (n_sys * nvar_per_sys * num_comm_bufs +
          buff_size * num_intermediate_buf) *
         sizeof(Float);
}

// ----------------------------------------------------------------------------
// Adapter function implementations
// ----------------------------------------------------------------------------

size_t pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const double *ds, const double *dl,
    const double *d, const double *du, const double *dw, const double *x,
    const int *t_dims, int t_ndim, int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(
      handle, ds, dl, d, du, dw, x, t_dims, static_cast<size_t>(t_ndim),
      t_solvedim);
}

size_t pentadsolver_D_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const double *ds, const double *dl,
    const double *d, const double *du, const double *dw, double *x,
    const int *t_dims, int t_ndim, int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(handle, ds, dl, d, du, dw, x,
                                                 t_dims, t_ndim, t_solvedim);
}

size_t pentadsolver_gpsv_batch_buffer_size_ext(pentadsolver_handle_t handle,
                                               const float *ds, const float *dl,
                                               const float *d, const float *du,
                                               const float *dw, const float *x,
                                               const int *t_dims, int t_ndim,
                                               int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(
      handle, ds, dl, d, du, dw, x, t_dims, static_cast<size_t>(t_ndim),
      t_solvedim);
}

size_t pentadsolver_S_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const float *ds, const float *dl,
    const float *d, const float *du, const float *dw, float *x,
    const int *t_dims, int t_ndim, int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(handle, ds, dl, d, du, dw, x,
                                                 t_dims, t_ndim, t_solvedim);
}

void pentadsolver_gpsv_batch(pentadsolver_handle_t handle, const double *ds,
                             const double *dl, const double *d,
                             const double *du, const double *dw, double *x,
                             const int *t_dims, int t_ndim, int t_solvedim,
                             void *t_buffer) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims,
                          static_cast<size_t>(t_ndim), t_solvedim, t_buffer);
}

void pentadsolver_D_gpsv_batch(pentadsolver_handle_t handle, const double *ds,
                               const double *dl, const double *d,
                               const double *du, const double *dw, double *x,
                               const int *t_dims, int t_ndim, int t_solvedim,
                               void *t_buffer) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims, t_ndim,
                          t_solvedim, t_buffer);
}

void pentadsolver_gpsv_batch(pentadsolver_handle_t handle, const float *ds,
                             const float *dl, const float *d, const float *du,
                             const float *dw, float *x, const int *t_dims,
                             int t_ndim, int t_solvedim, void *t_buffer) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims,
                          static_cast<size_t>(t_ndim), t_solvedim, t_buffer);
}

void pentadsolver_S_gpsv_batch(pentadsolver_handle_t handle, const float *ds,
                               const float *dl, const float *d, const float *du,
                               const float *dw, float *x, const int *t_dims,
                               int t_ndim, int t_solvedim, void *t_buffer) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims, t_ndim,
                          t_solvedim, t_buffer);
}
