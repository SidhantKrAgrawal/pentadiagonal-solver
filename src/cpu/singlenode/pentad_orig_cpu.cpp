#include <array>                   // for array
#include <cassert>                 // for assert
#include <cstddef>                 // for size_t
#include <functional>              // for multiplies
#include <numeric>                 // for accumulate
#include "pentadsolver.hpp"        // for pentadsolver_gpsv_batch, pentadso...
#include "pentadsolver_handle.hpp" // for pentadsolver_handle_t

template <typename Float>
void pentadsolver_x(const Float *ds, const Float *dl, const Float *d,
                    const Float *du, const Float *dw, Float *x,
                    size_t t_sys_size) {
  constexpr size_t N_MAX = 1024; // FIXME move to parameter, define
  std::array<Float, N_MAX> du2{};
  std::array<Float, N_MAX> dw2{};
  std::array<Float, N_MAX> x2{};
  // row 1 - normalise -- ds, dl 0
  du2[0] = du[0] / d[0];
  dw2[0] = dw[0] / d[0];
  x2[0]  = x[0] / d[0];

  // row 2 - /-1 , normalise - ds 0
  Float ddl = dl[1];
  Float dd  = d[1] - ddl * du2[0];
  du2[1]    = (du[1] - ddl * dw2[0]) / dd;
  dw2[1]    = dw[1] / dd;
  x2[1]     = (x[1] - ddl * x2[0]) / dd;

  // rest
  for (size_t i = 2; i < t_sys_size; ++i) {
    // row i - (dds*row{i-2}) - ddu'-row{i-1}, normalise
    // TODO: check with Istvan -- ds, dl, du, dw requirements -- last elements
    // must be 0, but are they accessible? remove comp du2[N-1], dw2[N-2..N-1]
    Float dds = ds[i];
    ddl       = dl[i] - dds * du2[i - 2];
    dd        = d[i] - dds * dw2[i - 2] - ddl * du2[i - 1];
    du2[i]    = (du[i] - ddl * dw2[i - 1]) / dd;
    dw2[i]    = dw[i] / dd;
    x2[i]     = (x[i] - dds * x2[i - 2] - ddl * x2[i - 1]) / dd;
  }
  //
  // Backward substitution
  //
  // row t_sys_size - 1
  x[t_sys_size - 1] = x2[t_sys_size - 1];
  // row t_sys_size - 2
  Float ddu         = du2[t_sys_size - 2];
  x[t_sys_size - 2] = x2[t_sys_size - 2] - ddu * x[t_sys_size - 1];
  // rest
  for (int i = static_cast<int>(t_sys_size) - 3; i >= 0; --i) {
    // row i - (ddw*row{i+2}) - ddu-row{i+1}
    Float ddw = dw2[i];
    ddu       = du2[i];
    x[i]      = x2[i] - ddw * x[i + 2] - ddu * x[i + 1];
  }
}

template <typename Float>
void pentadsolver_batch_x(const Float *ds, const Float *dl, const Float *d,
                          const Float *du, const Float *dw, Float *x,
                          size_t t_n_sys, size_t t_sys_size) {
  assert(t_sys_size > 4); // NOLINT

#pragma omp parallel for
  for (int i = 0; i < t_n_sys; ++i) {
    size_t sys_start = i * t_sys_size;
    pentadsolver_x(ds + sys_start, dl + sys_start, d + sys_start,
                   du + sys_start, dw + sys_start, x + sys_start, t_sys_size);
  }

}

template <typename Float>
void pentadsolver_strided(const Float *ds, const Float *dl, const Float *d,
                          const Float *du, const Float *dw, Float *x,
                          size_t t_sys_size, size_t t_stride) {
  constexpr size_t N_MAX = 1024; // FIXME move to parameter, define
  std::array<Float, N_MAX> du2{};
  std::array<Float, N_MAX> dw2{};
  std::array<Float, N_MAX> x2{};
  // row 1 - normalise -- ds, dl 0
  du2[0] = du[0 * t_stride] / d[0 * t_stride];
  dw2[0] = dw[0 * t_stride] / d[0 * t_stride];
  x2[0]  = x[0 * t_stride] / d[0 * t_stride];

  // row 2 - /-1 , normalise - ds 0
  Float ddl = dl[1 * t_stride];
  Float dd  = d[1 * t_stride] - ddl * du2[0];
  du2[1]    = (du[1 * t_stride] - ddl * dw2[0]) / dd;
  dw2[1]    = dw[1 * t_stride] / dd;
  x2[1]     = (x[1 * t_stride] - ddl * x2[0]) / dd;

  // rest
  for (size_t i = 2; i < t_sys_size; ++i) {
    // row i - (dds*row{i-2}) - ddu'-row{i-1}, normalise
    // TODO: check with Istvan -- ds, dl, du, dw requirements -- last elements
    // must be 0, but are they accessible? remove comp du2[N-1], dw2[N-2..N-1]
    Float dds = ds[i * t_stride];
    ddl       = dl[i * t_stride] - dds * du2[i - 2];
    dd        = d[i * t_stride] - dds * dw2[i - 2] - ddl * du2[i - 1];
    du2[i]    = (du[i * t_stride] - ddl * dw2[i - 1]) / dd;
    dw2[i]    = dw[i * t_stride] / dd;
    x2[i]     = (x[i * t_stride] - dds * x2[i - 2] - ddl * x2[i - 1]) / dd;
  }
  //
  // Backward substitution
  //
  // row t_sys_size - 1
  x[(t_sys_size - 1) * t_stride] = x2[t_sys_size - 1];
  // row t_sys_size - 2
  Float ddu = du2[t_sys_size - 2];
  x[(t_sys_size - 2) * t_stride] =
      x2[t_sys_size - 2] - ddu * x[(t_sys_size - 1) * t_stride];
  // rest
  for (int i = static_cast<int>(t_sys_size) - 3; i >= 0; --i) {
    // row i - (ddw*row{i+2}) - ddu-row{i+1}
    Float ddw = dw2[i];
    ddu       = du2[i];
    x[(i)*t_stride] =
        x2[i] - ddw * x[(i + 2) * t_stride] - ddu * x[(i + 1) * t_stride];
  }
}

template <typename Float>
void pentadsolver_batch_outermost(const Float *ds, const Float *dl,
                                  const Float *d, const Float *du,
                                  const Float *dw, Float *x, size_t t_n_sys,
                                  size_t t_sys_size) {
  assert(t_sys_size > 4); // NOLINT

#pragma omp parallel for
  for (int i = 0; i < t_n_sys; ++i) {
    size_t sys_start = i;
    pentadsolver_strided(ds + sys_start, dl + sys_start, d + sys_start,
                         du + sys_start, dw + sys_start, x + sys_start,
                         t_sys_size, t_n_sys);
  }
}

template <typename Float>
void pentadsolver_batch_middle(const Float *ds, const Float *dl, const Float *d,
                               const Float *du, const Float *dw, Float *x,
                               size_t t_n_sys_in, size_t t_sys_size,
                               size_t t_n_sys_out) {
  assert(t_sys_size > 4); // NOLINT

#pragma omp parallel for collapse(2)
  for (int i = 0; i < t_n_sys_out; ++i) {
    for (int j = 0; j < t_n_sys_in; ++j) {
      size_t sys_start = i * t_n_sys_in * t_sys_size + j;
      pentadsolver_strided(ds + sys_start, dl + sys_start, d + sys_start,
                           du + sys_start, dw + sys_start, x + sys_start,
                           t_sys_size, t_n_sys_in);
    }
  }
}

template <typename Float>
void pentadsolver_gpsv_batch_x(const Float *ds, const Float *dl, const Float *d,
                               const Float *du, const Float *dw, Float *x,
                               const int *t_dims, size_t t_ndims,
                               void * /*t_buffer*/) {
  size_t n_sys =
      std::accumulate(t_dims + 1, t_dims + t_ndims, 1, std::multiplies<>());
  size_t sys_size = t_dims[0];
  pentadsolver_batch_x(ds, dl, d, du, dw, x, n_sys, sys_size);

}

template <typename Float>
void pentadsolver_gpsv_batch_outermost(const Float *ds, const Float *dl,
                                       const Float *d, const Float *du,
                                       const Float *dw, Float *x,
                                       const int *t_dims, size_t t_ndims,
                                       void * /*t_buffer*/) {
  size_t n_sys =
      std::accumulate(t_dims, t_dims + t_ndims - 1, 1, std::multiplies<>());
  size_t sys_size = t_dims[t_ndims - 1];
  pentadsolver_batch_outermost(ds, dl, d, du, dw, x, n_sys, sys_size);
}

template <typename Float>
void pentadsolver_gpsv_batch_middle(const Float *ds, const Float *dl,
                                    const Float *d, const Float *du,
                                    const Float *dw, Float *x,
                                    const int *t_dims, size_t t_ndims,
                                    int t_solvedim, void * /*t_buffer*/) {
  size_t n_sys_in =
      std::accumulate(t_dims, t_dims + t_solvedim, 1, std::multiplies<>());
  size_t n_sys_out = std::accumulate(t_dims + t_solvedim + 1, t_dims + t_ndims,
                                     1, std::multiplies<>());
  size_t sys_size  = t_dims[t_solvedim];
  pentadsolver_batch_middle(ds, dl, d, du, dw, x, n_sys_in, sys_size,
                            n_sys_out);
}

template <typename Float>
void pentadsolver_gpsv_batch(const Float *ds, const Float *dl, const Float *d,
                             const Float *du, const Float *dw, Float *x,
                             const int *t_dims, size_t t_ndims, int t_solvedim,
                             void *t_buffer) {

  if (t_solvedim == 0) {
    pentadsolver_gpsv_batch_x(ds, dl, d, du, dw, x, t_dims, t_ndims, t_buffer);
  } else if (t_solvedim == t_ndims - 1) {
    pentadsolver_gpsv_batch_outermost(ds, dl, d, du, dw, x, t_dims, t_ndims,
                                      t_buffer);
  } else {
    pentadsolver_gpsv_batch_middle(ds, dl, d, du, dw, x, t_dims, t_ndims,
                                   t_solvedim, t_buffer);
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
    const Float * /*dw*/, const Float * /*x*/, const int * /*t_dims*/,
    size_t /*t_ndims*/, int /*t_solvedim*/) {
  return 0;
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
