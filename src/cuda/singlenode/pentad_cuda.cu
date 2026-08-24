#include <cooperative_groups.h> // for this_grid, grid_group
#include <cassert>              // for assert
#include <cstddef>              // for size_t
#include <cstdio>               // for fprintf (launch diagnostics)
#include <cstdlib>              // for getenv
#include <cstring>              // for strcmp
#include <functional>           // for multiplies
#include <numeric>              // for accumulate
#include "pentadsolver.hpp"     // for pentadsolver_gpsv_batch

// Scratch arrays du2/dw2/x2 live in caller-supplied global memory, laid out
// element-major/thread-minor as scratch_base[element_idx * n_threads + tid],
// so a warp accesses consecutive addresses at a fixed element index.

constexpr int X_TILE = 32; // tile width for transpose kernels

// Consume and test the status of the most recent kernel launch, clearing the
// sticky error so a fallback kernel starts clean.
//
// An opt-in kernel that cannot launch here (usually block size x register
// count over the per-block register file) must not be reported as having
// handled the solve: the dispatch would skip its fallback and leave the system
// unsolved, which looks like a very fast but wrong result.  PENTA_DEBUG_LAUNCH=1
// reports why a kernel declined, since a fallback otherwise looks the same as
// that kernel simply being faster.
static bool launch_succeeded(const char *what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    const char *dbg = std::getenv("PENTA_DEBUG_LAUNCH");
    if (dbg != nullptr && dbg[0] == '1') {
      std::fprintf(stderr, "[penta] %s launch declined: %s -- falling back\n",
                   what, cudaGetErrorString(err));
    }
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Algorithm 1 (Naive, code name "naive"): thread-per-system pentadiagonal
// Thomas solve, no coalescing.  The baseline, and the default when no
// algorithm is named.
// ---------------------------------------------------------------------------

template <typename Float>
__device__ void
pentadsolver_x(const Float *__restrict__ ds, const Float *__restrict__ dl,
               const Float *__restrict__ d, const Float *__restrict__ du,
               const Float *__restrict__ dw, Float *__restrict__ x,
               Float *__restrict__ du2, Float *__restrict__ dw2,
               Float *__restrict__ x2, size_t t_scratch_stride,
               size_t t_sys_size) {
  // row 1 - normalise -- ds, dl 0
  du2[0]                  = du[0] / d[0];
  dw2[0]                  = dw[0] / d[0];
  x2[0]                   = x[0] / d[0];

  // row 2 - /-1 , normalise - ds 0
  Float ddl = dl[1];
  Float dd  = d[1] - ddl * du2[0];
  du2[1 * t_scratch_stride] = (du[1] - ddl * dw2[0]) / dd;
  dw2[1 * t_scratch_stride] = dw[1] / dd;
  x2[1 * t_scratch_stride]  = (x[1] - ddl * x2[0]) / dd;

  // rest
  for (size_t i = 2; i < t_sys_size; ++i) {
    // row i - (dds*row{i-2}) - ddu'-row{i-1}, normalise
    // TODO: check with Istvan -- ds, dl, du, dw requirements -- last elements
    // must be 0, but are they accessible? remove comp du2[N-1], dw2[N-2..N-1]
    Float dds = ds[i];
    ddl = dl[i] - dds * du2[(i - 2) * t_scratch_stride];
    dd  = d[i] - dds * dw2[(i - 2) * t_scratch_stride]
              - ddl  * du2[(i - 1) * t_scratch_stride];
    du2[i * t_scratch_stride] = (du[i] - ddl * dw2[(i - 1) * t_scratch_stride]) / dd;
    dw2[i * t_scratch_stride] = dw[i] / dd;
    x2[i * t_scratch_stride]  = (x[i] - dds * x2[(i - 2) * t_scratch_stride]
                                       - ddl * x2[(i - 1) * t_scratch_stride]) / dd;
  }
  //
  // Backward substitution
  //
  // row t_sys_size - 1
  x[t_sys_size - 1] = x2[(t_sys_size - 1) * t_scratch_stride];
  // row t_sys_size - 2
  Float ddu         = du2[(t_sys_size - 2) * t_scratch_stride];
  x[t_sys_size - 2] = x2[(t_sys_size - 2) * t_scratch_stride]
                      - ddu * x[t_sys_size - 1];
  // rest
  for (int i = static_cast<int>(t_sys_size) - 3; i >= 0; --i) {
    // row i - (ddw*row{i+2}) - ddu-row{i+1}
    Float ddw = dw2[i * t_scratch_stride];
    ddu       = du2[i * t_scratch_stride];
    x[i]      = x2[i * t_scratch_stride] - ddw * x[i + 2] - ddu * x[i + 1];
  }
}

template <typename Float>
__global__ void pentadsolver_batch_x_kernel(const Float *__restrict__ ds,
                                            const Float *__restrict__ dl,
                                            const Float *__restrict__ d,
                                            const Float *__restrict__ du,
                                            const Float *__restrict__ dw,
                                            Float *__restrict__ x,
                                            Float *__restrict__ du2_g,
                                            Float *__restrict__ dw2_g,
                                            Float *__restrict__ x2_g,
                                            size_t t_n_sys, size_t t_sys_size) {
  size_t tid = cooperative_groups::this_grid().thread_rank();
  if (tid < t_n_sys) {
    size_t data_off = tid * t_sys_size;
    // Coalesced scratch: base = tid, stride = t_n_sys
    pentadsolver_x(ds + data_off, dl + data_off, d + data_off,
                   du + data_off, dw + data_off, x + data_off,
                   du2_g + tid, dw2_g + tid, x2_g + tid,
                   t_n_sys, t_sys_size);
  }
}

// Forward declaration, defined below after pentadsolver_strided.
template <typename Float>
__global__ void pentadsolver_batch_outermost_kernel(
    const Float *__restrict__ ds, const Float *__restrict__ dl,
    const Float *__restrict__ d, const Float *__restrict__ du,
    const Float *__restrict__ dw, Float *__restrict__ x,
    Float *__restrict__ du2_g, Float *__restrict__ dw2_g,
    Float *__restrict__ x2_g, size_t t_n_sys, size_t t_sys_size);

// Forward transpose: 3 arrays at a time from [sys][elem] to [elem][sys].
// Grid: (ceil(sys_size/X_TILE), ceil(n_sys/X_TILE)), x covers elems, y covers sys.
// Load is coalesced (consecutive tx at same ty read consecutive elements of same
// system).  Store is coalesced (consecutive tx at same ty write consecutive sys
// indices at same element) after swapping tx/ty for the output address.
template <typename Float>
__global__ void transpose_fwd_3(const Float *__restrict__ a_in,
                                 const Float *__restrict__ b_in,
                                 const Float *__restrict__ c_in,
                                 Float *__restrict__ a_out,
                                 Float *__restrict__ b_out,
                                 Float *__restrict__ c_out,
                                 size_t n_sys, size_t sys_size) {
  __shared__ Float tile[3][X_TILE][X_TILE + 1]; // +1 avoids bank conflicts

  int sys  = static_cast<int>(blockIdx.y) * X_TILE + threadIdx.y;
  int elem = static_cast<int>(blockIdx.x) * X_TILE + threadIdx.x;

  if (sys < static_cast<int>(n_sys) && elem < static_cast<int>(sys_size)) {
    size_t off                        = static_cast<size_t>(sys) * sys_size + elem;
    tile[0][threadIdx.y][threadIdx.x] = a_in[off];
    tile[1][threadIdx.y][threadIdx.x] = b_in[off];
    tile[2][threadIdx.y][threadIdx.x] = c_in[off];
  }
  __syncthreads();

  // Swap tx/ty so the write covers consecutive sys indices (coalesced)
  sys  = static_cast<int>(blockIdx.y) * X_TILE + threadIdx.x;
  elem = static_cast<int>(blockIdx.x) * X_TILE + threadIdx.y;

  if (sys < static_cast<int>(n_sys) && elem < static_cast<int>(sys_size)) {
    size_t off = static_cast<size_t>(elem) * n_sys + sys;
    a_out[off] = tile[0][threadIdx.x][threadIdx.y];
    b_out[off] = tile[1][threadIdx.x][threadIdx.y];
    c_out[off] = tile[2][threadIdx.x][threadIdx.y];
  }
}

// Backward transpose: 1 array from [elem][sys] to [sys][elem].
// Grid: (ceil(n_sys/X_TILE), ceil(sys_size/X_TILE)).
template <typename Float>
__global__ void transpose_bwd(const Float *__restrict__ x_in,
                               Float *__restrict__ x_out,
                               size_t n_sys, size_t sys_size) {
  __shared__ Float tile[X_TILE][X_TILE + 1];

  int sys  = static_cast<int>(blockIdx.x) * X_TILE + threadIdx.x;
  int elem = static_cast<int>(blockIdx.y) * X_TILE + threadIdx.y;

  if (sys < static_cast<int>(n_sys) && elem < static_cast<int>(sys_size))
    tile[threadIdx.y][threadIdx.x] =
        x_in[static_cast<size_t>(elem) * n_sys + sys];
  __syncthreads();

  sys  = static_cast<int>(blockIdx.x) * X_TILE + threadIdx.y;
  elem = static_cast<int>(blockIdx.y) * X_TILE + threadIdx.x;

  if (sys < static_cast<int>(n_sys) && elem < static_cast<int>(sys_size))
    x_out[static_cast<size_t>(sys) * sys_size + elem] =
        tile[threadIdx.x][threadIdx.y];
}

// ---------------------------------------------------------------------------
// Algorithm 2 (Global-Transpose, code name "transpose"): coalesced x-solve via
// explicit DRAM transpose.  Transpose 6 arrays to [elem][sys], solve with the
// strided kernel, transpose x back.  Scratch layout (9 regions of
// n_sys*sys_size each):
//   [0] du2  [1] dw2  [2] x2   [3] ds_T  [4] dl_T  [5] d_T
//   [6] du_T  [7] dw_T  [8] x_T
// ---------------------------------------------------------------------------
template <typename Float>
void pentadsolver_batch_x_coalesced(const Float *ds, const Float *dl,
                                    const Float *d, const Float *du,
                                    const Float *dw, Float *x,
                                    size_t t_n_sys, size_t t_sys_size,
                                    void *t_scratch) {
  size_t region = t_n_sys * t_sys_size;
  auto  *base   = static_cast<Float *>(t_scratch);
  Float *du2_g  = base + 0 * region;
  Float *dw2_g  = base + 1 * region;
  Float *x2_g   = base + 2 * region;
  Float *ds_T   = base + 3 * region;
  Float *dl_T   = base + 4 * region;
  Float *d_T    = base + 5 * region;
  Float *du_T   = base + 6 * region;
  Float *dw_T   = base + 7 * region;
  Float *x_T    = base + 8 * region;

  // Forward transpose: [sys][elem] → [elem][sys], 3 arrays per pass
  dim3 block(X_TILE, X_TILE);
  dim3 grid_fwd(static_cast<unsigned>((t_sys_size + X_TILE - 1) / X_TILE),
                static_cast<unsigned>((t_n_sys   + X_TILE - 1) / X_TILE));
  transpose_fwd_3<<<grid_fwd, block>>>(ds, dl, d,  ds_T, dl_T, d_T,  t_n_sys, t_sys_size);
  transpose_fwd_3<<<grid_fwd, block>>>(du, dw, x,  du_T, dw_T, x_T,  t_n_sys, t_sys_size);

  // Solve on transposed data, identical path to outermost (z-solve) kernel
  constexpr int block_dim = 128;
  int nblocks = 1 + (static_cast<int>(t_n_sys) - 1) / block_dim;
  pentadsolver_batch_outermost_kernel<<<nblocks, block_dim>>>(
      ds_T, dl_T, d_T, du_T, dw_T, x_T,
      du2_g, dw2_g, x2_g, t_n_sys, t_sys_size);

  // Backward transpose: [elem][sys] → [sys][elem] for x only
  dim3 grid_bwd(static_cast<unsigned>((t_n_sys   + X_TILE - 1) / X_TILE),
                static_cast<unsigned>((t_sys_size + X_TILE - 1) / X_TILE));
  transpose_bwd<<<grid_bwd, block>>>(x_T, x, t_n_sys, t_sys_size);
}

// ---------------------------------------------------------------------------
// Algorithm 3 (Hybrid Thomas-PCR, SPIKE partitioning, code name "thomas-pcr").
// One warp per system; lane t owns a contiguous M-element tile.
//
// The pentadiagonal generalisation of the Oxford tridsolver's hybrid
// Thomas-PCR: Thomas locally, PCR on the interface system.  A pentadiagonal
// cut has rank 2 rather than rank 1, so each tile exports four interface
// unknowns instead of two and the reduced system stays block-tridiagonal with
// 4x4 blocks instead of collapsing to a scalar 3-term recurrence.
//
// The method works at the solution level, not the coefficient level.  The
// solution on a tile is linear in its 4 boundary unknowns:
//   x_t = x_p + a*v_L1 + b*v_L2 + g*v_R1 + h*v_R2
// where x_p and the four spike vectors solve the local block A_t with zero
// carry-in (inter-tile couplings are stripped from A_t and moved to the spike
// right-hand sides).  There is no cross-lane dependency in the local
// elimination, so all lanes stay active throughout.  Working at the solution
// level is what makes this parallel: the Thomas coefficient recurrence is
// non-linear in its carry-in and admits no associative scan, but superposition
// does apply to the solution.
//
// Phase 2: local LU of A_t + 5 simultaneous solves.
// Phase 3: reduced block-tridiagonal system on w_t = (x[0],x[1],x[M-2],x[M-1]),
//          w_t = p_t + Lh_t*w_{t-1} + Rh_t*w_{t+1}, solved by block PCR with
//          one 4x4 Gauss-Jordan solve per lane per round.  Lh reads only the
//          tail pair of w_{t-1} and Rh only the head pair of w_{t+1}, so both
//          are stored as 4x2 blocks.
// Phase 4: pointwise correction from the stored spike vectors.
// ---------------------------------------------------------------------------

constexpr int ALGO3_M       = 8;    // default elements per thread (tile size)
constexpr int ALGO3_W       = 32;   // threads per system (must == warpSize)
constexpr int ALGO3_SYSSIZE = 256;  // default (legacy) system size (= ALGO3_M * ALGO3_W)
constexpr int ALGO3_BSYS    = 2;    // systems per thread block (= warps/block)
static_assert(ALGO3_M * ALGO3_W == ALGO3_SYSSIZE, "");

// The kernel is templated on M (elements per lane), giving sys_size = 32*M.
// Only three things depend on M, and all hold for any M >= 4: the tile-local
// index bounds (the 6 inter-tile coupling scalars always live on rows 0,1 and
// M-2,M-1), the boundary vector w = (x[0],x[1],x[M-2],x[M-1]) having 4 distinct
// rows, and the length of the local Thomas sweeps.  The reduced-system block
// PCR runs over the 32 lanes, so it is independent of M.  Instantiated for
// M = 4/8/10/12/16 -> sys_size 128/256/320/384/512.

// One round of the reduced-system block PCR.
//
// State per lane is w = p + L*(tail pair of w_{lane-s}) + R*(head pair of
// w_{lane+s}); a round doubles s by eliminating the neighbours at distance s,
// which requires solving a 4x4 system G*X = V.
//
// WANT_SPIKES=false peels the last round: afterwards only p is read, so the
// four spike columns of V and their elimination are dead and the compiler
// removes them (~112 of the round's ~220 FP64 slots).
//
// G updates are restricted to columns c > k: pivots run 0..3 in order and G is
// discarded on exit, so columns <= k are never read again (~36 slots/round).
template <typename Float, bool WANT_SPIKES, int L>
__device__ __forceinline__ void pop_pcr_round(
    int s, int lane, unsigned mask,
    Float (&p)[4], Float (&Lh)[4][2], Float (&R)[4][2]) {
  // `lane` is the SYSTEM-LOCAL lane (0..L-1).  The shuffles pass width=L so
  // each group of L lanes exchanges only within its own system.
  // From left neighbor (lane - s): tail pair of p, tail rows of Lh and R.
  Float pm2  = __shfl_up_sync(mask, p[2], s, L);
  Float pm3  = __shfl_up_sync(mask, p[3], s, L);
  Float Rm20 = __shfl_up_sync(mask, R[2][0], s, L);
  Float Rm21 = __shfl_up_sync(mask, R[2][1], s, L);
  Float Rm30 = __shfl_up_sync(mask, R[3][0], s, L);
  Float Rm31 = __shfl_up_sync(mask, R[3][1], s, L);
  // From right neighbor (lane + s): head pair of p, head rows of Lh and R.
  Float pq0  = __shfl_down_sync(mask, p[0], s, L);
  Float pq1  = __shfl_down_sync(mask, p[1], s, L);
  Float Lq00 = __shfl_down_sync(mask, Lh[0][0], s, L);
  Float Lq01 = __shfl_down_sync(mask, Lh[0][1], s, L);
  Float Lq10 = __shfl_down_sync(mask, Lh[1][0], s, L);
  Float Lq11 = __shfl_down_sync(mask, Lh[1][1], s, L);
  // Only the new spike columns need these four.
  Float Lm20 = Float(0), Lm21 = Float(0), Lm30 = Float(0), Lm31 = Float(0);
  Float Rq00 = Float(0), Rq01 = Float(0), Rq10 = Float(0), Rq11 = Float(0);
  if (WANT_SPIKES) {
    Lm20 = __shfl_up_sync(mask, Lh[2][0], s, L);
    Lm21 = __shfl_up_sync(mask, Lh[2][1], s, L);
    Lm30 = __shfl_up_sync(mask, Lh[3][0], s, L);
    Lm31 = __shfl_up_sync(mask, Lh[3][1], s, L);
    Rq00 = __shfl_down_sync(mask, R[0][0], s, L);
    Rq01 = __shfl_down_sync(mask, R[0][1], s, L);
    Rq10 = __shfl_down_sync(mask, R[1][0], s, L);
    Rq11 = __shfl_down_sync(mask, R[1][1], s, L);
  }
  if (lane < s) {          // no left neighbor
    pm2 = pm3 = Float(0);
    Lm20 = Lm21 = Lm30 = Lm31 = Float(0);
    Rm20 = Rm21 = Rm30 = Rm31 = Float(0);
  }
  if (lane + s >= L) {     // no right neighbor
    pq0 = pq1 = Float(0);
    Lq00 = Lq01 = Lq10 = Lq11 = Float(0);
    Rq00 = Rq01 = Rq10 = Rq11 = Float(0);
  }

  // G = I - Lh*Rm(tail rows) [cols 0,1] - Rh*Lq(head rows) [cols 2,3]
  // V = [ p' | newL | newR ] with
  //   p'   = p + Lh*(pm2,pm3) + Rh*(pq0,pq1)
  //   newL = Lh * Lm(tail rows),  newR = Rh * Rq(head rows)
  constexpr int NV = WANT_SPIKES ? 5 : 1;
  Float G[4][4], V[4][NV];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    G[i][0] = ((i == 0) ? Float(1) : Float(0)) - (Lh[i][0] * Rm20 + Lh[i][1] * Rm30);
    G[i][1] = ((i == 1) ? Float(1) : Float(0)) - (Lh[i][0] * Rm21 + Lh[i][1] * Rm31);
    G[i][2] = ((i == 2) ? Float(1) : Float(0)) - (R[i][0] * Lq00 + R[i][1] * Lq10);
    G[i][3] = ((i == 3) ? Float(1) : Float(0)) - (R[i][0] * Lq01 + R[i][1] * Lq11);
    V[i][0] = p[i] + Lh[i][0] * pm2 + Lh[i][1] * pm3 + R[i][0] * pq0 + R[i][1] * pq1;
    if (WANT_SPIKES) {
      V[i][1] = Lh[i][0] * Lm20 + Lh[i][1] * Lm30;
      V[i][2] = Lh[i][0] * Lm21 + Lh[i][1] * Lm31;
      V[i][3] = R[i][0] * Rq00 + R[i][1] * Rq10;
      V[i][4] = R[i][0] * Rq01 + R[i][1] * Rq11;
    }
  }

  // Gauss-Jordan solve G * X = V (no pivoting: G = I - (spike-tail
  // products), strongly diagonally dominant for ADI-class matrices).
#pragma unroll
  for (int k = 0; k < 4; k++) {
    Float inv_gkk = Float(1) / G[k][k];
#pragma unroll
    for (int c = k + 1; c < 4; c++) G[k][c] *= inv_gkk;
#pragma unroll
    for (int c = 0; c < NV; c++) V[k][c] *= inv_gkk;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      if (i == k) continue;
      Float f = G[i][k];
#pragma unroll
      for (int c = k + 1; c < 4; c++) G[i][c] -= f * G[k][c];
#pragma unroll
      for (int c = 0; c < NV; c++) V[i][c] -= f * V[k][c];
    }
  }

#pragma unroll
  for (int i = 0; i < 4; i++) {
    p[i] = V[i][0];
    if (WANT_SPIKES) {
      Lh[i][0] = V[i][1];
      Lh[i][1] = V[i][2];
      R[i][0]  = V[i][3];
      R[i][1]  = V[i][4];
    }
  }
}

// Warp-level POP-PCR solve of ONE 32*M-element pentadiagonal system held in
// registers.  Shared by Algorithm 3's x-solve kernel below.
// On entry: lane t's register tile holds coefficient rows [t*M, (t+1)*M) and
// x_r = the RHS tile.  On exit: x_r = the SOLUTION tile.  The coefficient
// register arrays are modified (coupling entries stripped), callers must not
// reuse them.  All 32 lanes of the warp must call this together (shuffles).
// PEEL_LAST drops the dead spike columns from the final PCR round.  It removes
// arithmetic but needs a second inlined copy of the round, which costs
// registers.  That is a win only where the kernel is arithmetic-bound: it is
// enabled for the x kernel (FP64-issue-bound) and disabled for the strided
// kernels, which are memory-bound and register-capped by __launch_bounds__,
// enabling it there spilled 33 registers and cost 5% on the z solve.
template <typename Float, int M, int L = ALGO3_W, bool PEEL_LAST = true>
__device__ __forceinline__ void pop_pcr_warp_solve(
    int lane,
    Float (&ds_r)[M], Float (&dl_r)[M], Float (&d_r)[M],
    Float (&du_r)[M], Float (&dw_r)[M], Float (&x_r)[M])
{
  static_assert(L >= 4 && L <= ALGO3_W && (L & (L - 1)) == 0,
                "lanes-per-system must be a power of two in [4, 32]");
  const uint32_t mask = 0xffffffff;

  // Save inter-tile coupling scalars, then strip them from the local block.
  // (For lane 0 the left couplings and for lane 31 the right couplings are
  // already zero in the global arrays, same result.)
  Float cds0 = ds_r[0], cdl0 = dl_r[0], cds1 = ds_r[1];
  Float cdw6 = dw_r[M - 2], cdu7 = du_r[M - 1], cdw7 = dw_r[M - 1];
  ds_r[0] = Float(0); ds_r[1] = Float(0); dl_r[0] = Float(0);
  du_r[M - 1] = Float(0); dw_r[M - 2] = Float(0); dw_r[M - 1] = Float(0);

  // -----------------------------------------------------------------------
  // Phase 2: local LU (zero carry-in, exact) fused with 5 forward sweeps:
  //   x_p  : particular solution, RHS = x_r
  //   v_L1 : response to u[8t-2],  RHS = -cds0 * e0
  //   v_L2 : response to u[8t-1],  RHS = -cdl0 * e0 - cds1 * e1
  //   v_R1 : response to u[8t+8],  RHS = -cdw6 * e6 - cdu7 * e7
  //   v_R2 : response to u[8t+9],  RHS = -cdw7 * e7
  // All 32 lanes active; no cross-lane dependency in this phase.
  // -----------------------------------------------------------------------
  Float du2l[M], dw2l[M];
  Float x_p[M], v_L1[M], v_L2[M], v_R1[M], v_R2[M];
  {
    Float pu2 = Float(0), pu1 = Float(0), pw2 = Float(0), pw1 = Float(0);
    Float pp2 = Float(0), pp1 = Float(0);
    Float pa2 = Float(0), pa1 = Float(0);
    Float pb2 = Float(0), pb1 = Float(0);
    Float pc2 = Float(0), pc1 = Float(0);
    Float pd2 = Float(0), pd1 = Float(0);
#pragma unroll
    for (int j = 0; j < M; j++) {
      Float ddl   = dl_r[j] - ds_r[j] * pu2;
      Float dd    = d_r[j] - ds_r[j] * pw2 - ddl * pu1;
      Float inv_d = Float(1) / dd;
      du2l[j] = (du_r[j] - ddl * pw1) * inv_d;
      dw2l[j] = dw_r[j] * inv_d;

      Float ra = (j == 0) ? -cds0 : Float(0);
      Float rb = (j == 0) ? -cdl0 : (j == 1) ? -cds1 : Float(0);
      Float rc = (j == M - 2) ? -cdw6 : (j == M - 1) ? -cdu7 : Float(0);
      Float rd = (j == M - 1) ? -cdw7 : Float(0);

      x_p[j]  = (x_r[j] - ds_r[j] * pp2 - ddl * pp1) * inv_d;
      v_L1[j] = (ra     - ds_r[j] * pa2 - ddl * pa1) * inv_d;
      v_L2[j] = (rb     - ds_r[j] * pb2 - ddl * pb1) * inv_d;
      v_R1[j] = (rc     - ds_r[j] * pc2 - ddl * pc1) * inv_d;
      v_R2[j] = (rd     - ds_r[j] * pd2 - ddl * pd1) * inv_d;

      pp2 = pp1; pp1 = x_p[j];
      pa2 = pa1; pa1 = v_L1[j];
      pb2 = pb1; pb1 = v_L2[j];
      pc2 = pc1; pc1 = v_R1[j];
      pd2 = pd1; pd1 = v_R2[j];
      pu2 = pu1; pu1 = du2l[j];
      pw2 = pw1; pw1 = dw2l[j];
    }
    // Backward substitution for all 5 vectors.
    Float sp0 = Float(0), sp1 = Float(0);
    Float sa0 = Float(0), sa1 = Float(0);
    Float sb0 = Float(0), sb1 = Float(0);
    Float sc0 = Float(0), sc1 = Float(0);
    Float sd0 = Float(0), sd1 = Float(0);
#pragma unroll
    for (int j = M - 1; j >= 0; j--) {
      x_p[j]  -= du2l[j] * sp0 + dw2l[j] * sp1;
      v_L1[j] -= du2l[j] * sa0 + dw2l[j] * sa1;
      v_L2[j] -= du2l[j] * sb0 + dw2l[j] * sb1;
      v_R1[j] -= du2l[j] * sc0 + dw2l[j] * sc1;
      v_R2[j] -= du2l[j] * sd0 + dw2l[j] * sd1;
      sp1 = sp0; sp0 = x_p[j];
      sa1 = sa0; sa0 = v_L1[j];
      sb1 = sb0; sb0 = v_L2[j];
      sc1 = sc0; sc0 = v_R1[j];
      sd1 = sd0; sd0 = v_R2[j];
    }
  }

  // -----------------------------------------------------------------------
  // Phase 3: reduced system.  Per lane: w = (x[0], x[1], x[M-2], x[M-1]) and
  //   w = p + Lh * (tail pair of w_{lane-1}) + Rh * (head pair of w_{lane+1})
  // p[4]; Lh, Rh stored compactly as 4x2.  Block PCR, 5 rounds: after round
  // r the coupling distance is 2^(r+1); after 5 rounds it exceeds 32 lanes
  // and every equation reads w = p directly.
  // -----------------------------------------------------------------------
  Float p[4], Lmat[4][2], R[4][2];
  {
    const int rows[4] = {0, 1, M - 2, M - 1};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      p[i]    = x_p[rows[i]];
      Lmat[i][0] = v_L1[rows[i]];
      Lmat[i][1] = v_L2[rows[i]];
      R[i][0] = v_R1[rows[i]];
      R[i][1] = v_R2[rows[i]];
    }
  }

  // Rounds 1 .. log2(W)-1 carry the spike columns forward; the final round
  // (s = W/2) only needs p, so it is peeled and runs the cheap variant.
  // Deliberately NOT unrolled: each round inlines ~200 FP64 instructions, and
  // unrolling all log2(L) of them inflates register pressure enough to spill
  // the strided (y/z) kernels, which also carry a shared-memory tile.  Rolled
  // costs one backward branch per round and measured strictly better.
  const int last = PEEL_LAST ? (L / 2) : L;   // `last` excluded when peeling
  for (int s = 1; s < last; s <<= 1) {
    pop_pcr_round<Float, true, L>(s, lane, mask, p, Lmat, R);
  }
  if (PEEL_LAST) {
    pop_pcr_round<Float, false, L>(L / 2, lane, mask, p, Lmat, R);
  }

  // After 5 rounds every lane holds its solved boundary values: w = p.
  // Neighbor boundary values for the correction:
  Float a_ = __shfl_up_sync(mask, p[2], 1, L);    // tail0 of lane-1
  Float b_ = __shfl_up_sync(mask, p[3], 1, L);    // tail1 of lane-1
  Float g_ = __shfl_down_sync(mask, p[0], 1, L);  // head0 of lane+1
  Float h_ = __shfl_down_sync(mask, p[1], 1, L);  // head1 of lane+1
  if (lane == 0)     { a_ = Float(0); b_ = Float(0); }
  if (lane == L - 1) { g_ = Float(0); h_ = Float(0); }

  // -----------------------------------------------------------------------
  // Phase 4: correction (all lanes), solution replaces the RHS in x_r.
  // -----------------------------------------------------------------------
#pragma unroll
  for (int j = 0; j < M; j++) {
    x_r[j] = x_p[j] + a_ * v_L1[j] + b_ * v_L2[j] + g_ * v_R1[j] + h_ * v_R2[j];
  }
}

// L = lanes per system.  With L < 32 a warp holds 32/L independent systems
// side by side, each PCR round is over L lanes instead of 32, and the reduced
// solve, whose cost is a fixed ~200 FP64 slots per lane per round, is
// amortised over the M = sys_size/L elements each lane owns.  Halving L both
// removes a round and doubles M, so the PCR cost per element falls
// superlinearly.  That is the dominant term in FP64.
template <typename Float, int M, int L = ALGO3_W>
__global__ void pentadsolver_batch_x_algo3_kernel_M(
    const Float *__restrict__ ds, const Float *__restrict__ dl,
    const Float *__restrict__ d,  const Float *__restrict__ du,
    const Float *__restrict__ dw, Float *__restrict__ x,
    int n_sys)
{
  static_assert(M >= 4, "POP-PCR needs M>=4 so the 4 boundary rows are distinct");
  constexpr int SYSSIZE  = L * M;
  constexpr int SYS_PER_W = ALGO3_W / L;   // systems packed side by side per warp
  int lane     = threadIdx.x;              // 0..31 (warp lane)
  int wid      = threadIdx.y;              // 0..ALGO3_BSYS-1
  int lane_sys = lane & (L - 1);           // lane within its system
  int sub      = lane / L;                 // which system of the warp
  int sys_id   = (blockIdx.x * ALGO3_BSYS + wid) * SYS_PER_W + sub;
  // Every lane must reach the shuffles, so out-of-range lanes clamp to the
  // last valid system and discard their write rather than returning early.
  const bool active = (sys_id < n_sys);
  if (!active) { sys_id = n_sys - 1; }

  const int sys_base  = sys_id * SYSSIZE;
  const int tile_base = lane_sys * M;

  // -----------------------------------------------------------------------
  // Phase 1: coalesced load of 6 arrays x 8 elements into registers.
  // -----------------------------------------------------------------------
  Float ds_r[M], dl_r[M], d_r[M];
  Float du_r[M], dw_r[M], x_r[M];
#pragma unroll
  for (int j = 0; j < M; j++) {
    int g   = sys_base + tile_base + j;
    ds_r[j] = ds[g];
    dl_r[j] = dl[g];
    d_r[j]  = d[g];
    du_r[j] = du[g];
    dw_r[j] = dw[g];
    x_r[j]  = x[g];
  }

  // Phases 2-4 (local LU + spike solves + block-PCR + correction) are the
  // shared warp-solve, see pop_pcr_warp_solve above.  x_r: RHS in, solution out.
  // Peel the last round in FP64 only.  FP64 here is arithmetic-bound, so
  // trading registers for fewer instructions pays.  FP32 is already
  // memory-bound at ~93% of peak, where the same trade only adds register
  // pressure, it cost 17% on the GTX 1080's FP32 x-solve when applied there.
  pop_pcr_warp_solve<Float, M, L, (sizeof(Float) == 8)>(
      lane_sys, ds_r, dl_r, d_r, du_r, dw_r, x_r);

  // Phase 5: coalesced write-back.
  if (active) {
#pragma unroll
    for (int j = 0; j < M; j++) {
      x[sys_base + tile_base + j] = x_r[j];
    }
  }
}

template <typename Float, int M, int L = ALGO3_W>
static void pentadsolver_batch_x_algo3_launch(const Float *ds, const Float *dl,
                                              const Float *d, const Float *du,
                                              const Float *dw, Float *x,
                                              size_t n_sys) {
  constexpr int SYS_PER_BLOCK = ALGO3_BSYS * (ALGO3_W / L);
  dim3 block(ALGO3_W, ALGO3_BSYS);
  dim3 grid(static_cast<unsigned>(
      (static_cast<int>(n_sys) + SYS_PER_BLOCK - 1) / SYS_PER_BLOCK));
  pentadsolver_batch_x_algo3_kernel_M<Float, M, L><<<grid, block>>>(
      ds, dl, d, du, dw, x, static_cast<int>(n_sys));
}

// True if this GPU runs FP64 at the full-rate 1:2 ratio (a data-centre part)
// rather than the 1:32 or 1:64 of a consumer part.  CUDA exposes no query for
// the FP64:FP32 ratio, so this is an explicit list of the architectures that
// have it: P100, V100, A100, H100, B100.  Anything not listed -- including any
// future architecture -- is treated as reduced-rate, which keeps the shipped
// behaviour unchanged on hardware nobody here has measured.
//
// Queried once and cached; cudaGetDeviceProperties is far too slow to sit in a
// dispatch path.  A failed query is reported as reduced-rate, again so the
// fallback is "behave as before".
static bool device_has_full_rate_fp64() {
  static const bool cached = [] {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { return false; }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) { return false; }
    const int cc = prop.major * 10 + prop.minor;
    return cc == 60 || cc == 70 || cc == 80 || cc == 90 || cc == 100;
  }();
  return cached;
}

// Returns true if POP-PCR handled this sys_size, false if no template exists
// for it.
//
// L (lanes per system) trades two costs against each other:
//   a LARGER L gives each lane fewer elements (M = sys_size/L), so fewer
//     registers and no spilling, but the reduced solve runs log2(L) PCR rounds
//     and so does more arithmetic;
//   a SMALLER L does less arithmetic but holds more elements per lane, and past
//     roughly M=8 in FP64 that spills to local memory.
// Which cost dominates is a property of the MACHINE, not of the algorithm, so
// this is chosen per device rather than baked in:
//
//   FP32 -- memory-bound on every card measured, so the redundant arithmetic is
//     free and the widest partitioning wins.  L=32 everywhere.  Measured at
//     256^3, x: RTX 3050 2.254 ms, GTX 1080 2.727, V100 0.680, H100 0.288 --
//     fastest at L=32 on all four.
//   FP64, full-rate card -- arithmetic is nearly free here too, so avoiding the
//     spill is what matters and L=32 wins by a wide margin.  Measured at 256^3,
//     x: V100 8.793 (L=8) / 6.183 (L=16) / 1.466 (L=32); H100 1.305 / 0.795 /
//     0.592.  The old fixed default of 16 cost a V100 4.2x.
//   FP64, reduced-rate card -- arithmetic dominates, so the extra PCR round of
//     L=32 is not worth the registers it saves.  L=16.  Measured at 256^3, x:
//     GTX 1080 20.68 (L=8) / 12.59 (L=16) / 19.97 (L=32); RTX 3050 19.30 /
//     20.86 / 37.94.  16 is best on one and within 8% on the other; 32 is
//     roughly 2x worse on both.
//
// Overridable with PENTA_PCR_LANES for sweeping, which is what the measurement
// scripts use.
template <typename Float>
bool pentadsolver_batch_x_algo3(const Float *ds, const Float *dl,
                                const Float *d, const Float *du,
                                const Float *dw, Float *x, size_t n_sys,
                                size_t sys_size) {
  int lanes = ALGO3_W;
  if (sizeof(Float) == 8 && !device_has_full_rate_fp64()) { lanes = 16; }
  if (const char *e = std::getenv("PENTA_PCR_LANES")) {
    const int v = std::atoi(e);
    if (v == 8 || v == 16 || v == 32) { lanes = v; }
  }

  if (lanes == 16 && sys_size % 16 == 0) {
    switch (sys_size / 16) {                          // = M (elements per lane)
      case 8:  pentadsolver_batch_x_algo3_launch<Float, 8,  16>(ds, dl, d, du, dw, x, n_sys); return true;
      case 16: pentadsolver_batch_x_algo3_launch<Float, 16, 16>(ds, dl, d, du, dw, x, n_sys); return true;
      // M=20 (sys_size 320).  Without it a 320-long system falls through to
      // L=32, tripling the per-element reduced-solve cost (98.8 vs 38.4 FP64
      // slots/element).
      case 20: pentadsolver_batch_x_algo3_launch<Float, 20, 16>(ds, dl, d, du, dw, x, n_sys); return true;
      case 24: pentadsolver_batch_x_algo3_launch<Float, 24, 16>(ds, dl, d, du, dw, x, n_sys); return true;
      case 32: pentadsolver_batch_x_algo3_launch<Float, 32, 16>(ds, dl, d, du, dw, x, n_sys); return true;
      default: break;                                 // fall through to L=32
    }
  }
  if (lanes == 8 && sys_size % 8 == 0) {
    switch (sys_size / 8) {
      case 16: pentadsolver_batch_x_algo3_launch<Float, 16, 8>(ds, dl, d, du, dw, x, n_sys); return true;
      case 32: pentadsolver_batch_x_algo3_launch<Float, 32, 8>(ds, dl, d, du, dw, x, n_sys); return true;
      default: break;
    }
  }

  if (sys_size % ALGO3_W != 0) return false;          // must tile evenly over 32 lanes
  switch (sys_size / ALGO3_W) {                        // = M (elements per lane)
    case 4:  pentadsolver_batch_x_algo3_launch<Float, 4 >(ds, dl, d, du, dw, x, n_sys); return true;
    case 8:  pentadsolver_batch_x_algo3_launch<Float, 8 >(ds, dl, d, du, dw, x, n_sys); return true;
    case 10: pentadsolver_batch_x_algo3_launch<Float, 10>(ds, dl, d, du, dw, x, n_sys); return true;
    case 12: pentadsolver_batch_x_algo3_launch<Float, 12>(ds, dl, d, du, dw, x, n_sys); return true;
    case 16: pentadsolver_batch_x_algo3_launch<Float, 16>(ds, dl, d, du, dw, x, n_sys); return true;
    default: return false;                          // M not instantiated -> fall back
  }
}

// ---------------------------------------------------------------------------
// Algorithm 3 for the strided directions (y and z).
//
// Same mathematics as the x-solve above; what differs is the memory
// choreography.  A y- or z-system's elements are strided in DRAM (stride N for
// y, N^2 for z), so per-lane direct loads would be scatter reads.  Instead the
// block cooperatively stages each array through a shared-memory tile: thread
// tid fetches element (elem = idx/BSYS, sys = idx%BSYS), giving BSYS
// consecutive values per group and so full-sector coalesced reads; lane t of
// warp w then extracts its M elements from column w.  The +1 pad on the tile's
// minor dimension avoids the worst bank conflicts.
//
// y and z keep separate dispatch paths and block-size constants because the
// two strides want opposite things: y stays largely page-local, while every z
// element row lands on a different DRAM page.
// ---------------------------------------------------------------------------

// Threads per block for Algorithm 1's strided kernels.  The kernel needs ~107
// registers, so block size sets how many blocks fit per SM.  Swept at 256^3
// FP64: 64 -> y 9.60 / z 9.53, 128 -> 9.63 / 9.58, 256 -> 9.85 / 9.71,
// 512 -> 10.01 / 9.99.  64 wins, but by only ~0.3-0.5% and with a flat curve,
// the kernel is bandwidth-bound, not occupancy-bound.  1024 exceeds the
// per-block register file and is rejected at launch.
constexpr int ALGO1_Y_BLOCK = 64;
constexpr int ALGO1_Z_BLOCK = 64;

// The two constants above are defaults measured on one GPU, not properties of
// the algorithm, so they are overridable per machine: PENTA_Y_BLOCK and
// PENTA_Z_BLOCK.  This is a launch parameter, so any positive multiple of the
// warp size up to 1024 is legal, though a large block can exceed the per-block
// register file and be refused at launch.
static int env_block(const char *name, int fallback) {
  const char *e = std::getenv(name);
  if (e == nullptr || e[0] == '\0') { return fallback; }
  const int v = std::atoi(e);
  if (v <= 0 || v > 1024 || (v % 32) != 0) { return fallback; }
  return v;
}

// Systems (= warps) per block, tuned per direction.  At 256^3 FP32:
//   y (stride N)    BSYS=4 -> 3.21 ms, 8 -> 3.76, 16 -> 4.27.  Narrow blocks
//                   win: element rows stay page-local and occupancy stays high.
//   z (stride N^2)  BSYS=8 -> 4.78 ms, 16 -> 4.39, 32 -> 12.07 (spills).  Wide
//                   blocks win: each element row is on its own DRAM page, so
//                   read width is the only lever.  BSYS=16 reads 64 B not 32 B.
constexpr int ALGO3_Y_BSYS = 4;
constexpr int ALGO3_Z_BSYS = 16;

// Stage one array per sync round.  A single reused tile beat a paired
// two-tile scheme in measurement, and forcing occupancy with
// __launch_bounds__ caused spills that cost more than the concurrency won.
template <typename Float, int M, int BSYS>
__device__ __forceinline__ void algo3_strided_stage_in(
    const Float *__restrict__ src, size_t base0, int stride, int n_sys,
    int s0, int tid, int lane, int w, bool sys_valid,
    Float (&tile)[ALGO3_W * M][BSYS + 1], Float (&r)[M]) {
#pragma unroll
  for (int k = 0; k < M; k++) {
    int idx  = k * (ALGO3_W * BSYS) + tid; // 0 .. SYSSIZE*BSYS - 1
    int elem = idx / BSYS;
    int so   = idx % BSYS;
    if (s0 + so < n_sys) {
      tile[elem][so] = src[base0 + (size_t)so + (size_t)elem * (size_t)stride];
    }
  }
  __syncthreads();
  if (sys_valid) {
#pragma unroll
    for (int j = 0; j < M; j++) { r[j] = tile[lane * M + j][w]; }
  }
  __syncthreads(); // tile is reused for the next array
}

template <typename Float, int M, int BSYS>
__device__ __forceinline__ void pentadsolver_strided_algo3_body(
    const Float *__restrict__ ds, const Float *__restrict__ dl,
    const Float *__restrict__ d,  const Float *__restrict__ du,
    const Float *__restrict__ dw, Float *__restrict__ x,
    int n_sys, int stride, int run_len)
{
  static_assert(M >= 4, "POP-PCR needs M>=4 so the 4 boundary rows are distinct");
  constexpr int SYSSIZE = ALGO3_W * M;
  __shared__ Float tile[SYSSIZE][BSYS + 1];

  const int  lane = threadIdx.x;             // 0..31
  const int  w    = threadIdx.y;             // 0..BSYS-1 (local system id)
  const int  tid  = w * ALGO3_W + lane;      // flat id, for cooperative I/O
  const int  s0   = blockIdx.x * BSYS;
  const int  sys  = s0 + w;
  const bool sys_valid = (sys < n_sys);

  // System s lives at base(s) = (s/run_len)*(run_len*SYSSIZE) + (s%run_len),
  // element j at base(s) + j*stride.  run_len % BSYS == 0 and s0 % BSYS == 0
  // guarantee the block's systems share one contiguous run, so
  // base(s0 + k) = base(s0) + k.
  const size_t base0 = (size_t)(s0 / run_len) * ((size_t)run_len * SYSSIZE) +
                       (size_t)(s0 % run_len);

  Float ds_r[M], dl_r[M], d_r[M], du_r[M], dw_r[M], x_r[M];
  algo3_strided_stage_in<Float, M, BSYS>(ds, base0, stride, n_sys, s0, tid,
                                         lane, w, sys_valid, tile, ds_r);
  algo3_strided_stage_in<Float, M, BSYS>(dl, base0, stride, n_sys, s0, tid,
                                         lane, w, sys_valid, tile, dl_r);
  algo3_strided_stage_in<Float, M, BSYS>(d, base0, stride, n_sys, s0, tid,
                                         lane, w, sys_valid, tile, d_r);
  algo3_strided_stage_in<Float, M, BSYS>(du, base0, stride, n_sys, s0, tid,
                                         lane, w, sys_valid, tile, du_r);
  algo3_strided_stage_in<Float, M, BSYS>(dw, base0, stride, n_sys, s0, tid,
                                         lane, w, sys_valid, tile, dw_r);
  algo3_strided_stage_in<Float, M, BSYS>(x, base0, stride, n_sys, s0, tid,
                                         lane, w, sys_valid, tile, x_r);

  if (sys_valid) {
    // PEEL_LAST=false: this path is memory-bound and register-capped, so the
    // peeled round's extra registers cost more than its saved arithmetic.
    pop_pcr_warp_solve<Float, M, ALGO3_W, false>(lane, ds_r, dl_r, d_r, du_r,
                                                 dw_r, x_r);
  }

  // Staged write-back (reverse of stage_in).
  if (sys_valid) {
#pragma unroll
    for (int j = 0; j < M; j++) { tile[lane * M + j][w] = x_r[j]; }
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < M; k++) {
    int idx  = k * (ALGO3_W * BSYS) + tid;
    int elem = idx / BSYS;
    int so   = idx % BSYS;
    if (s0 + so < n_sys) {
      x[base0 + (size_t)so + (size_t)elem * (size_t)stride] = tile[elem][so];
    }
  }
}

// Two wrappers around the same body, differing only in whether ptxas is told
// the block size.  The bound is needed only for wide blocks: at BSYS=16 the
// kernel's 124 registers round up to the 128-register granule, so a 512-thread
// block claims the whole 64K per-block register file and the launch is
// rejected.  __launch_bounds__ sheds enough registers to keep it launchable,
// which buys the wide DRAM reads z needs.  It is not applied to narrow blocks,
// where it measured 21% slower on y (3.89 vs 3.21 ms at 256^3 FP32).
template <typename Float, int M, int BSYS>
__global__ void pentadsolver_batch_strided_algo3_kernel(
    const Float *__restrict__ ds, const Float *__restrict__ dl,
    const Float *__restrict__ d,  const Float *__restrict__ du,
    const Float *__restrict__ dw, Float *__restrict__ x,
    int n_sys, int stride, int run_len) {
  pentadsolver_strided_algo3_body<Float, M, BSYS>(ds, dl, d, du, dw, x, n_sys,
                                                  stride, run_len);
}

template <typename Float, int M, int BSYS>
__global__ __launch_bounds__(ALGO3_W *BSYS) void
pentadsolver_batch_strided_algo3_kernel_bounded(
    const Float *__restrict__ ds, const Float *__restrict__ dl,
    const Float *__restrict__ d,  const Float *__restrict__ du,
    const Float *__restrict__ dw, Float *__restrict__ x,
    int n_sys, int stride, int run_len) {
  pentadsolver_strided_algo3_body<Float, M, BSYS>(ds, dl, d, du, dw, x, n_sys,
                                                  stride, run_len);
}

// Blocks at or above this width need the register cap to launch at all.
constexpr int ALGO3_BOUNDED_BSYS = 16;

template <typename Float, int M, int BSYS>
static bool algo3_strided_launch(const Float *ds, const Float *dl,
                                 const Float *d, const Float *du,
                                 const Float *dw, Float *x, size_t n_sys,
                                 size_t stride, size_t run_len) {
  // Only instantiate variants whose smem tile fits the 48KB limit.  48KB is the
  // hard ceiling on STATICALLY declared shared memory on every architecture --
  // the larger 96KB (Volta) and 227KB (Hopper) budgets are reachable only by
  // dynamic shared memory, which this tile is not.  So this bound stays, and a
  // system too large for the requested BSYS is handled by retrying a narrower
  // block in pentadsolver_batch_strided_algo3 rather than by raising the limit.
  constexpr size_t tile_bytes =
      (size_t)(ALGO3_W * M) * (BSYS + 1) * sizeof(Float);
  if constexpr (tile_bytes <= 48 * 1024) {
    dim3 block(ALGO3_W, BSYS);
    dim3 grid(static_cast<unsigned>((n_sys + BSYS - 1) / BSYS));
    if constexpr (BSYS >= ALGO3_BOUNDED_BSYS) {
      pentadsolver_batch_strided_algo3_kernel_bounded<Float, M, BSYS>
          <<<grid, block>>>(ds, dl, d, du, dw, x, static_cast<int>(n_sys),
                            static_cast<int>(stride),
                            static_cast<int>(run_len));
    } else {
      pentadsolver_batch_strided_algo3_kernel<Float, M, BSYS><<<grid, block>>>(
          ds, dl, d, du, dw, x, static_cast<int>(n_sys),
          static_cast<int>(stride), static_cast<int>(run_len));
    }
    // A launch can still be rejected at runtime even though the shared-memory
    // tile fits, most often when BSYS pushes threads-per-block x registers
    // past the per-block register file.  Report failure so the caller falls
    // back to a working kernel instead of silently leaving the system
    // unsolved.
    return launch_succeeded("Algorithm 3 strided");
  } else {
    (void)ds; (void)dl; (void)d; (void)du; (void)dw; (void)x;
    (void)n_sys; (void)stride; (void)run_len;
    return false;
  }
}

// Returns true if Algorithm 3 handled this strided batch.
template <typename Float, int BSYS>
static bool algo3_strided_bsys(const Float *ds, const Float *dl, const Float *d,
                               const Float *du, const Float *dw, Float *x,
                               size_t n_sys, size_t sys_size, size_t stride,
                               size_t run_len) {
  if (run_len % BSYS != 0) { return false; }
  switch (sys_size / ALGO3_W) { // = M (elements per lane)
    case 4:  return algo3_strided_launch<Float, 4,  BSYS>(ds, dl, d, du, dw, x, n_sys, stride, run_len);
    case 8:  return algo3_strided_launch<Float, 8,  BSYS>(ds, dl, d, du, dw, x, n_sys, stride, run_len);
    case 10: return algo3_strided_launch<Float, 10, BSYS>(ds, dl, d, du, dw, x, n_sys, stride, run_len);
    case 12: return algo3_strided_launch<Float, 12, BSYS>(ds, dl, d, du, dw, x, n_sys, stride, run_len);
    case 16: return algo3_strided_launch<Float, 16, BSYS>(ds, dl, d, du, dw, x, n_sys, stride, run_len);
    default: return false; // M not instantiated
  }
}

template <typename Float>
static bool algo3_strided_dispatch_bsys(const Float *ds, const Float *dl,
                                        const Float *d, const Float *du,
                                        const Float *dw, Float *x,
                                        size_t n_sys, size_t sys_size,
                                        size_t stride, size_t run_len,
                                        int bsys) {
  switch (bsys) {
    case 2:  return algo3_strided_bsys<Float, 2 >(ds, dl, d, du, dw, x, n_sys, sys_size, stride, run_len);
    case 4:  return algo3_strided_bsys<Float, 4 >(ds, dl, d, du, dw, x, n_sys, sys_size, stride, run_len);
    case 8:  return algo3_strided_bsys<Float, 8 >(ds, dl, d, du, dw, x, n_sys, sys_size, stride, run_len);
    case 16: return algo3_strided_bsys<Float, 16>(ds, dl, d, du, dw, x, n_sys, sys_size, stride, run_len);
    case 32: return algo3_strided_bsys<Float, 32>(ds, dl, d, du, dw, x, n_sys, sys_size, stride, run_len);
    default: return false; // BSYS not instantiated
  }
}

// Instantiated BSYS values, widest first.
constexpr int ALGO3_BSYS_CHOICES[] = {32, 16, 8, 4, 2};

// BSYS (systems per block) is a template parameter, so the set of usable values
// is fixed at compile time, but which of them is fastest is a property of the
// GPU rather than of the algorithm.  The default comes from the calling
// direction; PENTA_Y_BSYS and PENTA_Z_BSYS select another instantiated value so
// the knob can be swept on whatever machine this is.
//
// If the preferred width cannot run, retry the narrower ones before giving up.
// BSYS sets the shared tile size -- 32*M*(BSYS+1)*sizeof(Float) -- against a
// 48KB static limit, so the widest setting is the first to become unusable as
// the system grows.  At 384^3 in FP64 the z default of BSYS=16 needs 52,224 B
// and is refused, while BSYS=8 needs 27,648 B and runs; without this retry the
// whole solve declined and the run was recorded as a failure, which is what
// happened on both a V100 and an H100.  Narrowing also relaxes the
// run_len % BSYS divisibility test, never tightens it.
//
// A narrower block is slower, so the substitution is reported once rather than
// made silently: an unannounced fallback reads as a real measurement of the
// requested configuration.
template <typename Float>
static bool pentadsolver_batch_strided_algo3(const Float *ds, const Float *dl,
                                             const Float *d, const Float *du,
                                             const Float *dw, Float *x,
                                             size_t n_sys, size_t sys_size,
                                             size_t stride, size_t run_len,
                                             int bsys) {
  if (sys_size % ALGO3_W != 0) { return false; }
  if (algo3_strided_dispatch_bsys<Float>(ds, dl, d, du, dw, x, n_sys, sys_size,
                                         stride, run_len, bsys)) {
    return true;
  }
  for (const int cand : ALGO3_BSYS_CHOICES) {
    if (cand >= bsys) { continue; }
    if (algo3_strided_dispatch_bsys<Float>(ds, dl, d, du, dw, x, n_sys, sys_size,
                                           stride, run_len, cand)) {
      static bool announced = false;
      if (!announced) {
        announced = true;
        std::fprintf(stderr,
                     "[penta] Algorithm 3 strided: systems-per-block %d could not "
                     "run at system size %zu (shared tile over the 48KB limit, or "
                     "a rejected launch); using %d instead, which is slower\n",
                     bsys, sys_size, cand);
      }
      return true;
    }
  }
  return false;
}

// Resolve the systems-per-block for one strided direction.  Only instantiated
// values are accepted; anything else keeps the default rather than failing a
// solve over a tuning knob.
static int env_bsys(const char *name, int fallback) {
  const char *e = std::getenv(name);
  if (e == nullptr || e[0] == '\0') { return fallback; }
  const int v = std::atoi(e);
  if (v == 2 || v == 4 || v == 8 || v == 16 || v == 32) { return v; }
  return fallback;
}

// ---------------------------------------------------------------------------
// Shared tile-staging helper for the x-direction (used by Algorithm 4 below).
//
// An x-direction solve reads along the contiguous axis, so a thread-per-system
// kernel would issue lane-strided loads and amplify DRAM traffic ~4x.  This
// helper removes that penalty without the explicit out-of-place transposes
// Algorithm 2 pays 14 extra array passes for: a warp cooperatively reads
// 32 systems x 8 elements in fully coalesced 64-byte segments and un-transposes
// ownership through a shared-memory tile, so each thread leaves with its own
// system's 8 elements in registers.
// ---------------------------------------------------------------------------

constexpr int XT_C    = 8;  // elements per chunk (64B FP64 segment per system)
constexpr int XT_SYS  = 32; // systems per warp (one per lane)
constexpr int XT_BSYS = 8;  // warps per block

// Cooperative transposed load of a 32-system x 8-element tile of `src` into
// per-thread registers r[0..7] (thread `lane` receives system sys0+lane).
// Lanes are grouped 8-per-system so each group reads one contiguous 64-byte
// segment; the smem tile un-transposes ownership.  Guarded for n_sys tails.
template <typename Float>
__device__ __forceinline__ void fused_transpose_load_tile(
    const Float *__restrict__ src, Float (*tile)[XT_C + 1], Float *r,
    int sys0, int lane, int i0, int sys_size, int n_sys) {
#pragma unroll
  for (int p = 0; p < XT_C; p++) {
    int row = p * 4 + lane / XT_C;  // system row within the tile
    int col = lane % XT_C;          // element within the chunk
    int sys = sys0 + row;
    tile[row][col] =
        (sys < n_sys) ? src[(size_t)sys * sys_size + i0 + col] : Float(0);
  }
  __syncwarp();
#pragma unroll
  for (int j = 0; j < XT_C; j++) r[j] = tile[lane][j];
  __syncwarp();
}

// ---------------------------------------------------------------------------
// Algorithm 4: ADI-structure solver, precomputed shared factorization.
// Restricted problem class, selected only by naming shared-fact explicitly.
//
// In ADI applications every line in a direction shares the same pentadiagonal
// coefficients and only the RHS differs.  The Thomas factorization then
// depends on the element index alone, so it can be computed once per direction
// (a serial sys_size-long recurrence) and broadcast to every system.  Per
// direction this removes the 5 coefficient arrays from the traffic (the table
// is 5*sys_size elements, L2-resident, read by every warp at the same index),
// removes the du2/dw2 scratch, and removes the per-row divides.
//
// Traffic drops to 4 array passes (b in, x2 out, x2 in, x out) against 13 for
// the general path, a 3.25x lower floor.
//
// The table is built from system 0's coefficients.  Callers must guarantee all
// systems share them; this is NOT checked at runtime.
// ---------------------------------------------------------------------------

// Serial one-thread factorization of system 0 into the table
// tab[5*sys_size] = [ds | ddl | inv_dd | du2 | dw2].  Zero-seeded carries
// reproduce the first-row special cases exactly: out-of-range coupling values
// multiply zero carries, so boundary garbage is harmless.
template <typename Float>
__global__ void pentadsolver_algo4_build_table_kernel(
    const Float *__restrict__ ds, const Float *__restrict__ dl,
    const Float *__restrict__ d, const Float *__restrict__ du,
    const Float *__restrict__ dw, Float *__restrict__ tab, int sys_size,
    size_t stride) {
  if (threadIdx.x != 0 || blockIdx.x != 0) { return; }
  Float pu2 = Float(0), pu1 = Float(0), pw2 = Float(0), pw1 = Float(0);
  for (int i = 0; i < sys_size; i++) {
    Float dsv  = ds[(size_t)i * stride];
    Float ddl  = dl[(size_t)i * stride] - dsv * pu2;
    Float dd   = d[(size_t)i * stride] - dsv * pw2 - ddl * pu1;
    Float inv  = Float(1) / dd;
    Float du2v = (du[(size_t)i * stride] - ddl * pw1) * inv;
    Float dw2v = dw[(size_t)i * stride] * inv;
    tab[i]                = dsv;
    tab[sys_size + i]     = ddl;
    tab[2 * sys_size + i] = inv;
    tab[3 * sys_size + i] = du2v;
    tab[4 * sys_size + i] = dw2v;
    pu2 = pu1; pu1 = du2v;
    pw2 = pw1; pw1 = dw2v;
  }
}

// x-direction Algorithm 4: Algorithm 4's fused-transpose staging, but only b and x
// move through the tiles; coefficients come from the broadcast table and
// only x2 goes through scratch.
template <typename Float>
__global__ void pentadsolver_batch_x_algo4_kernel(
    const Float *__restrict__ tab, Float *__restrict__ x,
    Float *__restrict__ x2_g, int n_sys, int sys_size)
{
  __shared__ Float tile_s[XT_BSYS][XT_SYS][XT_C + 1];

  int lane = threadIdx.x;
  int wid  = threadIdx.y;
  int sys0 = (blockIdx.x * XT_BSYS + wid) * XT_SYS;
  if (sys0 >= n_sys) return;
  Float(*tile)[XT_C + 1] = tile_s[wid];

  int  sys    = sys0 + lane;
  bool active = sys < n_sys;

  const Float *tds  = tab;
  const Float *tddl = tab + sys_size;
  const Float *tinv = tab + 2 * sys_size;
  const Float *tdu2 = tab + 3 * sys_size;
  const Float *tdw2 = tab + 4 * sys_size;

  // Forward sweep: transpose-load b, table-broadcast coefficients, stream
  // only x2 to global scratch (coalesced [i * n_sys + sys]).
  Float px2 = Float(0), px1 = Float(0);
  for (int i0 = 0; i0 < sys_size; i0 += XT_C) {
    Float b_r[XT_C];
    fused_transpose_load_tile(x, tile, b_r, sys0, lane, i0, sys_size, n_sys);
    Float c_ds[XT_C], c_ddl[XT_C], c_inv[XT_C];
#pragma unroll
    for (int j = 0; j < XT_C; j++) { // same address across the warp: broadcast
      c_ds[j]  = tds[i0 + j];
      c_ddl[j] = tddl[i0 + j];
      c_inv[j] = tinv[i0 + j];
    }
    if (active) {
#pragma unroll
      for (int j = 0; j < XT_C; j++) {
        Float x2v = (b_r[j] - c_ds[j] * px2 - c_ddl[j] * px1) * c_inv[j];
        px2 = px1; px1 = x2v;
        x2_g[(size_t)(i0 + j) * n_sys + sys] = x2v;
      }
    }
  }

  // Backward substitution: x2 scratch + table, transpose-store x.
  Float s0 = Float(0), s1 = Float(0);
  for (int i0 = sys_size - XT_C; i0 >= 0; i0 -= XT_C) {
    Float c_du2[XT_C], c_dw2[XT_C];
#pragma unroll
    for (int j = 0; j < XT_C; j++) {
      c_du2[j] = tdu2[i0 + j];
      c_dw2[j] = tdw2[i0 + j];
    }
    Float xo[XT_C];
#pragma unroll
    for (int j = 0; j < XT_C; j++) xo[j] = Float(0);
    if (active) {
#pragma unroll
      for (int j = XT_C - 1; j >= 0; j--) {
        size_t g  = (size_t)(i0 + j) * n_sys + sys;
        Float  xj = x2_g[g] - c_du2[j] * s0 - c_dw2[j] * s1;
        s1 = s0; s0 = xj; xo[j] = xj;
      }
    }
    __syncwarp();
#pragma unroll
    for (int j = 0; j < XT_C; j++) tile[lane][j] = xo[j];
    __syncwarp();
#pragma unroll
    for (int p = 0; p < XT_C; p++) {
      int row = p * 4 + lane / XT_C;
      int col = lane % XT_C;
      int s   = sys0 + row;
      if (s < n_sys) x[(size_t)s * sys_size + i0 + col] = tile[row][col];
    }
    __syncwarp();
  }
}

template <typename Float>
void pentadsolver_batch_x_algo4(const Float *ds, const Float *dl,
                                 const Float *d, const Float *du,
                                 const Float *dw, Float *x, size_t n_sys,
                                 size_t sys_size, void *t_scratch) {
  auto  *x2_g = static_cast<Float *>(t_scratch);
  Float *tab  = x2_g + n_sys * sys_size; // table after x2 (buffer holds 3x)
  pentadsolver_algo4_build_table_kernel<<<1, 1>>>(ds, dl, d, du, dw, tab,
                                                   static_cast<int>(sys_size),
                                                   /*stride=*/1);
  int  warps = (static_cast<int>(n_sys) + XT_SYS - 1) / XT_SYS;
  dim3 block(XT_SYS, XT_BSYS);
  dim3 grid(static_cast<unsigned>((warps + XT_BSYS - 1) / XT_BSYS));
  pentadsolver_batch_x_algo4_kernel<<<grid, block>>>(
      tab, x, x2_g, static_cast<int>(n_sys), static_cast<int>(sys_size));
}

// y/z-direction Algorithm 4: the legacy thread-per-system strided walk (already
// naturally coalesced across adjacent systems) with table-broadcast
// coefficients, register carries, and x2-only scratch.
template <typename Float>
__global__ void pentadsolver_batch_yz_algo4_kernel(
    const Float *__restrict__ tab, Float *__restrict__ x,
    Float *__restrict__ x2_g, size_t n_sys, size_t sys_size, size_t stride,
    size_t run_len) {
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_sys) { return; }
  const Float *tds  = tab;
  const Float *tddl = tab + sys_size;
  const Float *tinv = tab + 2 * sys_size;
  const Float *tdu2 = tab + 3 * sys_size;
  const Float *tdw2 = tab + 4 * sys_size;

  size_t base = (tid / run_len) * (run_len * sys_size) + (tid % run_len);
  Float *b  = x + base;                 // RHS in, solution out (in place)
  Float *x2 = x2_g + tid;               // coalesced scratch, stride n_sys

  Float px2 = Float(0), px1 = Float(0);
  for (size_t i = 0; i < sys_size; i++) {
    Float x2v = (b[i * stride] - tds[i] * px2 - tddl[i] * px1) * tinv[i];
    px2 = px1; px1 = x2v;
    x2[i * n_sys] = x2v;
  }
  Float s0 = Float(0), s1 = Float(0);
  for (int i = static_cast<int>(sys_size) - 1; i >= 0; i--) {
    Float xj = x2[(size_t)i * n_sys] - tdu2[i] * s0 - tdw2[i] * s1;
    s1 = s0; s0 = xj;
    b[(size_t)i * stride] = xj;
  }
}

// ---------------------------------------------------------------------------
// Algorithm selection
// ---------------------------------------------------------------------------
// There is no automatic dispatch.  The kernel for each direction is named
// explicitly, or the hardcoded default (naive, Algorithm 1) runs.
//
//   PENTA_ALGO                   sets all three directions at once.
//   PENTA_XALGO / PENTA_YALGO /  set one direction each, overriding PENTA_ALGO.
//   PENTA_ZALGO
//
// x accepts   naive | transpose | thomas-pcr | shared-fact
// y, z accept naive | thomas-pcr | shared-fact   (transpose is x only: y and z
//             are already coalesced, so transposing them is pure added cost)
//
// An unrecognised name, a name the direction does not accept, or a size the
// requested kernel has no template for is a hard error: the run stops rather
// than quietly solving with a different kernel and mislabelling the result.

enum class StridedDir { Middle, Outermost };   // y, z

// ---------------------------------------------------------------------------
// Executed-kernel record
// ---------------------------------------------------------------------------
// Each dispatch site stamps the name it launched, and the benchmark app records
// that rather than the name that was requested.  Index: 0 = x, 1 = y, 2 = z.
namespace {
const char *g_ran_kernel[3] = {"none", "none", "none"};
void        note_kernel(int dir, const char *name) { g_ran_kernel[dir] = name; }
} // namespace

extern "C" const char *pentadsolver_kernel_that_ran(int dir) {
  return (dir >= 0 && dir < 3) ? g_ran_kernel[dir] : "none";
}

// Stop the run with a diagnostic.  Reached only for a selector the build cannot
// honour, which is a configuration mistake rather than a runtime condition.
[[noreturn]] static void penta_fail(const char *dir, const char *algo,
                                    const char *why) {
  std::fprintf(stderr, "[penta] %s solve: cannot run '%s': %s\n", dir, algo, why);
  std::exit(EXIT_FAILURE);
}

// The selector for one direction.  A per-direction variable wins over
// PENTA_ALGO; with neither set the default is naive.
static const char *algo_selector(int dir) {
  static const char *names[3] = {"PENTA_XALGO", "PENTA_YALGO", "PENTA_ZALGO"};
  const char *specific = std::getenv(names[dir]);
  if (specific != nullptr && specific[0] != '\0') { return specific; }
  const char *all = std::getenv("PENTA_ALGO");
  if (all != nullptr && all[0] != '\0') { return all; }
  return "naive";
}

static const char *strided_algo_selector(StridedDir dir) {
  return algo_selector(dir == StridedDir::Middle ? 1 : 2);
}

static bool strided_algo_is(StridedDir dir, const char *name) {
  return std::strcmp(strided_algo_selector(dir), name) == 0;
}

// Returns true if Algorithm 4 handled this strided batch.
template <typename Float>
bool pentadsolver_batch_yz_algo4(const Float *ds, const Float *dl,
                                  const Float *d, const Float *du,
                                  const Float *dw, Float *x, size_t n_sys,
                                  size_t sys_size, size_t stride,
                                  size_t run_len, void *t_scratch,
                                  StridedDir dir) {
  if (!strided_algo_is(dir, "shared-fact")) { return false; }
  auto  *x2_g = static_cast<Float *>(t_scratch);
  Float *tab  = x2_g + n_sys * sys_size;
  pentadsolver_algo4_build_table_kernel<<<1, 1>>>(
      ds, dl, d, du, dw, tab, static_cast<int>(sys_size), stride);
  constexpr int block_dim = 128;
  int nblocks = 1 + (static_cast<int>(n_sys) - 1) / block_dim;
  pentadsolver_batch_yz_algo4_kernel<<<nblocks, block_dim>>>(
      tab, x, x2_g, n_sys, sys_size, stride, run_len);
  return true;
}

// ---------------------------------------------------------------------------

template <typename Float>
void pentadsolver_batch_x(const Float *ds, const Float *dl, const Float *d,
                          const Float *du, const Float *dw, Float *x,
                          size_t t_n_sys, size_t t_sys_size, void *t_scratch) {
  assert(t_sys_size > 4); // NOLINT

  size_t scratch_elems = t_n_sys * t_sys_size;
  auto  *du2_g = static_cast<Float *>(t_scratch);
  auto  *dw2_g = du2_g + scratch_elems;
  auto  *x2_g  = dw2_g + scratch_elems;

  // Set up the execution configuration
  constexpr int block_dim_x = 128;
  int nblocks               = 1 + (static_cast<int>(t_n_sys) - 1) / block_dim_x;

  pentadsolver_batch_x_kernel<<<nblocks, block_dim_x>>>(ds, dl, d, du, dw, x,
                                                        du2_g, dw2_g, x2_g,
                                                        t_n_sys, t_sys_size);
}

template <typename Float>
__device__ void
pentadsolver_strided(const Float *__restrict__ ds, const Float *__restrict__ dl,
                     const Float *__restrict__ d, const Float *__restrict__ du,
                     const Float *__restrict__ dw, Float *__restrict__ x,
                     Float *__restrict__ du2, Float *__restrict__ dw2,
                     Float *__restrict__ x2, size_t t_scratch_stride,
                     size_t t_sys_size, size_t t_stride) {
  // row 1 - normalise -- ds, dl 0
  du2[0]                    = du[0 * t_stride] / d[0 * t_stride];
  dw2[0]                    = dw[0 * t_stride] / d[0 * t_stride];
  x2[0]                     = x[0 * t_stride] / d[0 * t_stride];

  // row 2 - /-1 , normalise - ds 0
  Float ddl = dl[1 * t_stride];
  Float dd  = d[1 * t_stride] - ddl * du2[0];
  du2[1 * t_scratch_stride] = (du[1 * t_stride] - ddl * dw2[0]) / dd;
  dw2[1 * t_scratch_stride] = dw[1 * t_stride] / dd;
  x2[1 * t_scratch_stride]  = (x[1 * t_stride] - ddl * x2[0]) / dd;

  // rest
  for (size_t i = 2; i < t_sys_size; ++i) {
    // row i - (dds*row{i-2}) - ddu'-row{i-1}, normalise
    // TODO: check with Istvan -- ds, dl, du, dw requirements -- last elements
    // must be 0, but are they accessible? remove comp du2[N-1], dw2[N-2..N-1]
    Float dds = ds[i * t_stride];
    ddl = dl[i * t_stride] - dds * du2[(i - 2) * t_scratch_stride];
    dd  = d[i * t_stride] - dds * dw2[(i - 2) * t_scratch_stride]
                          - ddl  * du2[(i - 1) * t_scratch_stride];
    du2[i * t_scratch_stride] = (du[i * t_stride] - ddl * dw2[(i - 1) * t_scratch_stride]) / dd;
    dw2[i * t_scratch_stride] = dw[i * t_stride] / dd;
    x2[i * t_scratch_stride]  = (x[i * t_stride] - dds * x2[(i - 2) * t_scratch_stride]
                                                  - ddl * x2[(i - 1) * t_scratch_stride]) / dd;
  }
  //
  // Backward substitution
  //
  // row t_sys_size - 1
  x[(t_sys_size - 1) * t_stride] = x2[(t_sys_size - 1) * t_scratch_stride];
  // row t_sys_size - 2
  Float ddu = du2[(t_sys_size - 2) * t_scratch_stride];
  x[(t_sys_size - 2) * t_stride] =
      x2[(t_sys_size - 2) * t_scratch_stride]
      - ddu * x[(t_sys_size - 1) * t_stride];
  // rest
  for (int i = static_cast<int>(t_sys_size) - 3; i >= 0; --i) {
    // row i - (ddw*row{i+2}) - ddu-row{i+1}
    Float ddw = dw2[i * t_scratch_stride];
    ddu       = du2[i * t_scratch_stride];
    x[(i)*t_stride] =
        x2[i * t_scratch_stride]
        - ddw * x[(i + 2) * t_stride] - ddu * x[(i + 1) * t_stride];
  }
}

template <typename Float>
__global__ void pentadsolver_batch_outermost_kernel(
    const Float *__restrict__ ds, const Float *__restrict__ dl,
    const Float *__restrict__ d, const Float *__restrict__ du,
    const Float *__restrict__ dw, Float *__restrict__ x,
    Float *__restrict__ du2_g, Float *__restrict__ dw2_g,
    Float *__restrict__ x2_g, size_t t_n_sys, size_t t_sys_size) {
  assert(t_sys_size > 4); // NOLINT
  size_t tid = cooperative_groups::this_grid().thread_rank();
  if (tid < t_n_sys) {
    size_t sys_start = tid;
    // Coalesced scratch: base = tid, stride = t_n_sys
    pentadsolver_strided(ds + sys_start, dl + sys_start, d + sys_start,
                         du + sys_start, dw + sys_start, x + sys_start,
                         du2_g + tid, dw2_g + tid, x2_g + tid,
                         t_n_sys, t_sys_size, t_n_sys);
  }
}

template <typename Float>
__global__ void pentadsolver_batch_middle_kernel(
    const Float *__restrict__ ds, const Float *__restrict__ dl,
    const Float *__restrict__ d, const Float *__restrict__ du,
    const Float *__restrict__ dw, Float *__restrict__ x,
    Float *__restrict__ du2_g, Float *__restrict__ dw2_g,
    Float *__restrict__ x2_g, size_t t_n_sys_in, size_t t_sys_size,
    size_t t_n_sys_out) {
  assert(t_sys_size > 4); // NOLINT
  size_t tid    = cooperative_groups::this_grid().thread_rank();
  size_t n_total = t_n_sys_in * t_n_sys_out;
  if (tid < n_total) {
    size_t i         = tid / t_n_sys_in;
    size_t j         = tid % t_n_sys_in;
    size_t sys_start = i * t_n_sys_in * t_sys_size + j;
    // Coalesced scratch: base = tid, stride = n_total
    pentadsolver_strided(ds + sys_start, dl + sys_start, d + sys_start,
                         du + sys_start, dw + sys_start, x + sys_start,
                         du2_g + tid, dw2_g + tid, x2_g + tid,
                         n_total, t_sys_size, t_n_sys_in);
  }
}

template <typename Float>
void pentadsolver_batch_outermost(const Float *__restrict__ ds,
                                  const Float *__restrict__ dl,
                                  const Float *__restrict__ d,
                                  const Float *__restrict__ du,
                                  const Float *__restrict__ dw,
                                  Float *__restrict__ x, size_t t_n_sys,
                                  size_t t_sys_size, void *t_scratch) {
  assert(t_sys_size > 4); // NOLINT

  // --- z direction (outermost): stride N^2, run length N^2 ----------------
  constexpr StridedDir DIR = StridedDir::Outermost;

  const char *sel = strided_algo_selector(DIR);

  // Algorithm 4: ADI-structure solver, restricted problem class.
  if (pentadsolver_batch_yz_algo4(ds, dl, d, du, dw, x, t_n_sys, t_sys_size,
                                   /*stride=*/t_n_sys, /*run_len=*/t_n_sys,
                                   t_scratch, DIR)) {
    note_kernel(2, "shared-fact");
    return;
  }

  // Algorithm 3: register-resident POP-PCR, no scratch.
  if (std::strcmp(sel, "thomas-pcr") == 0) {
    if (!pentadsolver_batch_strided_algo3<Float>(
            ds, dl, d, du, dw, x, t_n_sys, t_sys_size,
            /*stride=*/t_n_sys, /*run_len=*/t_n_sys,
            env_bsys("PENTA_Z_BSYS", ALGO3_Z_BSYS))) {
      penta_fail("z", sel,
                 "no template for this system size, or the shared tile exceeds "
                 "the per-block limit");
    }
    note_kernel(2, "thomas-pcr");
    return;
  }

  if (std::strcmp(sel, "naive") != 0) {
    penta_fail("z", sel, "expected naive, thomas-pcr or shared-fact "
                         "(transpose is an x-direction kernel only)");
  }

  size_t scratch_elems = t_n_sys * t_sys_size;
  auto  *du2_g = static_cast<Float *>(t_scratch);
  auto  *dw2_g = du2_g + scratch_elems;
  auto  *x2_g  = dw2_g + scratch_elems;

  // Set up the execution configuration
  note_kernel(2, "naive");
  const int block_dim_x = env_block("PENTA_Z_BLOCK", ALGO1_Z_BLOCK);
  int nblocks               = 1 + (static_cast<int>(t_n_sys) - 1) / block_dim_x;
  pentadsolver_batch_outermost_kernel<<<nblocks, block_dim_x>>>(
      ds, dl, d, du, dw, x, du2_g, dw2_g, x2_g, t_n_sys, t_sys_size);
}

template <typename Float>
void pentadsolver_batch_middle(const Float *__restrict__ ds,
                               const Float *__restrict__ dl,
                               const Float *__restrict__ d,
                               const Float *__restrict__ du,
                               const Float *__restrict__ dw,
                               Float *__restrict__ x, size_t t_n_sys_in,
                               size_t t_sys_size, size_t t_n_sys_out,
                               void *t_scratch) {
  assert(t_sys_size > 4); // NOLINT

  // --- y direction (middle): stride N, runs of N adjacent systems ---------
  constexpr StridedDir DIR = StridedDir::Middle;

  const char *sel = strided_algo_selector(DIR);

  // Algorithm 4: ADI-structure solver, restricted problem class.
  if (pentadsolver_batch_yz_algo4(ds, dl, d, du, dw, x,
                                   t_n_sys_in * t_n_sys_out, t_sys_size,
                                   /*stride=*/t_n_sys_in,
                                   /*run_len=*/t_n_sys_in, t_scratch, DIR)) {
    note_kernel(1, "shared-fact");
    return;
  }

  // Algorithm 3: register-resident POP-PCR, no scratch.
  if (std::strcmp(sel, "thomas-pcr") == 0) {
    if (!pentadsolver_batch_strided_algo3<Float>(
            ds, dl, d, du, dw, x, t_n_sys_in * t_n_sys_out, t_sys_size,
            /*stride=*/t_n_sys_in, /*run_len=*/t_n_sys_in,
            env_bsys("PENTA_Y_BSYS", ALGO3_Y_BSYS))) {
      penta_fail("y", sel,
                 "no template for this system size, or the shared tile exceeds "
                 "the per-block limit");
    }
    note_kernel(1, "thomas-pcr");
    return;
  }

  if (std::strcmp(sel, "naive") != 0) {
    penta_fail("y", sel, "expected naive, thomas-pcr or shared-fact "
                         "(transpose is an x-direction kernel only)");
  }

  size_t n_total       = t_n_sys_in * t_n_sys_out;
  size_t scratch_elems = n_total * t_sys_size;
  auto  *du2_g = static_cast<Float *>(t_scratch);
  auto  *dw2_g = du2_g + scratch_elems;
  auto  *x2_g  = dw2_g + scratch_elems;

  // Set up the execution configuration
  note_kernel(1, "naive");
  const int block_dim_x = env_block("PENTA_Y_BLOCK", ALGO1_Y_BLOCK);
  int nblocks =
      1 + (static_cast<int>(n_total) - 1) / block_dim_x;
  pentadsolver_batch_middle_kernel<<<nblocks, block_dim_x>>>(
      ds, dl, d, du, dw, x, du2_g, dw2_g, x2_g, t_n_sys_in, t_sys_size,
      t_n_sys_out);
}

template <typename Float>
void pentadsolver_gpsv_batch_outermost(const Float *ds, const Float *dl,
                                       const Float *d, const Float *du,
                                       const Float *dw, Float *x,
                                       const int *t_dims, size_t t_ndims,
                                       void *t_buffer) {
  size_t n_sys =
      std::accumulate(t_dims, t_dims + t_ndims - 1, size_t(1), std::multiplies<>());
  size_t sys_size = t_dims[t_ndims - 1];
  pentadsolver_batch_outermost(ds, dl, d, du, dw, x, n_sys, sys_size, t_buffer);
}

template <typename Float>
void pentadsolver_gpsv_batch_middle(const Float *ds, const Float *dl,
                                    const Float *d, const Float *du,
                                    const Float *dw, Float *x,
                                    const int *t_dims, size_t t_ndims,
                                    int t_solvedim, void *t_buffer) {
  size_t n_sys_in =
      std::accumulate(t_dims, t_dims + t_solvedim, size_t(1), std::multiplies<>());
  size_t n_sys_out = std::accumulate(t_dims + t_solvedim + 1, t_dims + t_ndims,
                                     size_t(1), std::multiplies<>());
  size_t sys_size  = t_dims[t_solvedim];
  pentadsolver_batch_middle(ds, dl, d, du, dw, x, n_sys_in, sys_size,
                            n_sys_out, t_buffer);
}

template <typename Float>
void pentadsolver_gpsv_batch_x(const Float *ds, const Float *dl, const Float *d,
                               const Float *du, const Float *dw, Float *x,
                               const int *t_dims, size_t t_ndims,
                               void *t_buffer) {
  size_t n_sys    = std::accumulate(t_dims + 1, t_dims + t_ndims, size_t(1), std::multiplies<>());
  size_t sys_size = static_cast<size_t>(t_dims[0]);

  const char *sel = algo_selector(0);

  if (std::strcmp(sel, "naive") == 0) {
    note_kernel(0, "naive");
    pentadsolver_batch_x(ds, dl, d, du, dw, x, n_sys, sys_size, t_buffer);
    return;
  }
  if (std::strcmp(sel, "transpose") == 0) {
    note_kernel(0, "transpose");
    pentadsolver_batch_x_coalesced(ds, dl, d, du, dw, x, n_sys, sys_size,
                                   t_buffer);
    return;
  }
  if (std::strcmp(sel, "thomas-pcr") == 0) {
    if (!pentadsolver_batch_x_algo3(ds, dl, d, du, dw, x, n_sys, sys_size)) {
      penta_fail("x", sel,
                 "no template for this system size (instantiated for 128, 256, "
                 "320, 384 and 512)");
    }
    note_kernel(0, "thomas-pcr");
    return;
  }
  // Algorithm 4: ADI-structure solver, restricted problem class.
  if (std::strcmp(sel, "shared-fact") == 0) {
    if (sys_size % XT_C != 0) {
      penta_fail("x", sel, "system size must be a multiple of 8");
    }
    note_kernel(0, "shared-fact");
    pentadsolver_batch_x_algo4(ds, dl, d, du, dw, x, n_sys, sys_size, t_buffer);
    return;
  }
  penta_fail("x", sel,
             "expected naive, transpose, thomas-pcr or shared-fact");
}

template <typename Float>
void pentadsolver_gpsv_batch(const Float *ds, const Float *dl, const Float *d,
                             const Float *du, const Float *dw, Float *x,
                             const int *t_dims, size_t t_ndims, int t_solvedim,
                             void *t_buffer) {

  if (t_solvedim == 0) {
    pentadsolver_gpsv_batch_x(ds, dl, d, du, dw, x, t_dims, t_ndims, t_buffer);
  } else if (t_solvedim == static_cast<int>(t_ndims) - 1) {
    pentadsolver_gpsv_batch_outermost(ds, dl, d, du, dw, x, t_dims, t_ndims,
                                      t_buffer);
  } else {
    pentadsolver_gpsv_batch_middle(ds, dl, d, du, dw, x, t_dims, t_ndims,
                                   t_solvedim, t_buffer);
  }

  // A rejected launch otherwise leaves the system silently UNSOLVED, which
  // presents as an impossibly fast but completely wrong result.  The opt-in
  // kernels detect this themselves and fall back; the always-available ones
  // have no fallback, so surface it here rather than letting it pass.  This
  // consumes the sticky error either way, so a later unrelated call is not
  // blamed for it.
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                 "[penta] solvedim %d launch FAILED: %s, output is invalid\n",
                 t_solvedim, cudaGetErrorString(err));
  }
}

// ----------------------------------------------------------------------------
// Pentadsolver context functions
// ----------------------------------------------------------------------------

void pentadsolver_create(pentadsolver_handle_t * /*handle*/,
                         void * /*communicator*/, int /*ndims*/,
                         const int * /*num_procs*/) {}
void pentadsolver_destroy(pentadsolver_handle_t * /*handle*/) {}

// ----------------------------------------------------------------------------
// Buffer size calculation
// ----------------------------------------------------------------------------

template <typename Float>
[[nodiscard]] size_t pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t /*handle*/, const Float * /*ds*/,
    const Float * /*dl*/, const Float * /*d*/, const Float * /*du*/,
    const Float * /*dw*/, const Float * /*x*/, const int *t_dims,
    size_t t_ndims, int t_solvedim) {
  size_t n_total;
  size_t sys_size;
  if (t_solvedim == 0) {
    n_total  = std::accumulate(t_dims + 1, t_dims + t_ndims, size_t(1), std::multiplies<>());
    sys_size = static_cast<size_t>(t_dims[0]);
    // Algorithm 2: transpose path needs 9 scratch regions
    return 9 * n_total * sys_size * sizeof(Float);
  } else if (t_solvedim == static_cast<int>(t_ndims) - 1) {
    n_total  = std::accumulate(t_dims, t_dims + t_ndims - 1, size_t(1), std::multiplies<>());
    sys_size = static_cast<size_t>(t_dims[t_ndims - 1]);
  } else {
    n_total =
        std::accumulate(t_dims, t_dims + t_solvedim, size_t(1), std::multiplies<>()) *
        std::accumulate(t_dims + t_solvedim + 1, t_dims + t_ndims, size_t(1), std::multiplies<>());
    sys_size = static_cast<size_t>(t_dims[t_solvedim]);
  }
  return 3 * n_total * sys_size * sizeof(Float);
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

void pentadsolver_gpsv_batch(pentadsolver_handle_t /*handle*/, const double *ds,
                             const double *dl, const double *d,
                             const double *du, const double *dw, double *x,
                             const int *t_dims, int t_ndim, int t_solvedim,
                             void *t_buffer) {
  pentadsolver_gpsv_batch(ds, dl, d, du, dw, x, t_dims,
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

void pentadsolver_gpsv_batch(pentadsolver_handle_t /*handle*/, const float *ds,
                             const float *dl, const float *d, const float *du,
                             const float *dw, float *x, const int *t_dims,
                             int t_ndim, int t_solvedim, void *t_buffer) {
  pentadsolver_gpsv_batch(ds, dl, d, du, dw, x, t_dims,
                          static_cast<size_t>(t_ndim), t_solvedim, t_buffer);
}

void pentadsolver_S_gpsv_batch(pentadsolver_handle_t handle, const float *ds,
                               const float *dl, const float *d, const float *du,
                               const float *dw, float *x, const int *t_dims,
                               int t_ndim, int t_solvedim, void *t_buffer) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims, t_ndim,
                          t_solvedim, t_buffer);
}
