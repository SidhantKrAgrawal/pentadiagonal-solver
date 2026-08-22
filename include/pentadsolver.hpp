#ifndef PENTADSOLVER_HPP_INCLUDED
#define PENTADSOLVER_HPP_INCLUDED

// ----------------------------------------------------------------------------
// Internal Libraries and Headers
// ----------------------------------------------------------------------------
#include "pentadsolver_handle.hpp"

// ----------------------------------------------------------------------------
// Standard Libraries and Headers
// ----------------------------------------------------------------------------
#include <cstddef>

// ----------------------------------------------------------------------------
// Library defines
// ----------------------------------------------------------------------------
#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

// ----------------------------------------------------------------------------
// Pentadsolver context functions
// ----------------------------------------------------------------------------
/**
 * @brief This creates a pentadsolver context object. NOOP in single node
 * backends.
 *
 * @param[in] handle        Pentadsolver library context.
 * @param[in] communicator  MPI Cartesian communicator in MPI backends.
 * @param[in] ndims         Number of dimensions
 * @param[in] num_procs     Number of MPI processes along each dimension.
 *
 */
EXTERN_C
void pentadsolver_create(pentadsolver_handle_t *handle, void *communicator,
                         int ndims, const int *num_procs);

EXTERN_C
void pentadsolver_destroy(pentadsolver_handle_t *handle);

// ----------------------------------------------------------------------------
// Buffer size calculation
// ----------------------------------------------------------------------------
/**
 * @brief This function computes the amount of extra memory required for
 * pentadsolver_D_gpsv_batch() for the solution of a batch pentadiagonal system
 * of size t_dims, along the axis t_solvedim.
 *
 * @param[in] handle  Pentadsolver library context.
 * @param[in] ds      Array containing the 2nd lower diagonal.
 * @param[in] dl      Array containing the lower diagonal.
 * @param[in] d       Array containing the main diagonal.
 * @param[in] du      Array containing the upper diagonal.
 * @param[in] dw      Array containing the 2nd upper diagonal.
 * @param[in] x       Dense array of RHS as input
 * @param[in] t_dims      The dimensions of the LHS arrays.
 * @param[in] t_ndims     The length of the t_dims array.
 * @param[in] t_solvedim  The dimension along which the systems are formed.
 *
 * @return size of required buffers in bytes.
 */
EXTERN_C
[[nodiscard]] size_t pentadsolver_D_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const double *ds, const double *dl,
    const double *d, const double *du, const double *dw, const double *x,
    const int *t_dims, int t_ndim, int t_solvedim);
EXTERN_C
[[nodiscard]] size_t pentadsolver_S_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const float *ds, const float *dl,
    const float *d, const float *du, const float *dw, const float *x,
    const int *t_dims, int t_ndim, int t_solvedim);

#ifdef __cplusplus
[[nodiscard]] size_t pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const double *ds, const double *dl,
    const double *d, const double *du, const double *dw, const double *x,
    const int *t_dims, int t_ndim, int t_solvedim);
[[nodiscard]] size_t pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const float *ds, const float *dl,
    const float *d, const float *du, const float *dw, const float *x,
    const int *t_dims, int t_ndim, int t_solvedim);
#endif
// ----------------------------------------------------------------------------
// Solver functions
// ----------------------------------------------------------------------------
/**
 * @brief This function computes the solution of multiple penta-diagonal systems
 * along a specified axis.
 *
 *
 * @param[in] ds      Array containing the 2nd lower diagonal.
 * @param[in] dl      Array containing the lower diagonal.
 * @param[in] d       Array containing the main diagonal.
 * @param[in] du      Array containing the upper diagonal.
 * @param[in] dw      Array containing the 2nd upper diagonal.
 * @param[in,out] x   Dense array of RHS as input, dense solution array as
 * output
 * @param[in] t_dims      The dimensions of the LHS arrays.
 * @param[in] t_ndims     The length of the t_dims array.
 * @param[in] t_solvedim  The dimension along which the systems are formed.
 * @param[in] t_buffer  Buffer allocated by the user, with size at least
 * pentadsolver_D_gpsv_batch_buffer_size_ext()
 *
 */
EXTERN_C
void pentadsolver_D_gpsv_batch(pentadsolver_handle_t handle, const double *ds,
                               const double *dl, const double *d,
                               const double *du, const double *dw, double *x,
                               const int *t_dims, int t_ndim, int t_solvedim,
                               void *t_buffer);

EXTERN_C
void pentadsolver_S_gpsv_batch(pentadsolver_handle_t handle, const float *ds,
                               const float *dl, const float *d, const float *du,
                               const float *dw, float *x, const int *t_dims,
                               int t_ndim, int t_solvedim, void *t_buffer);

#ifdef __cplusplus
void pentadsolver_gpsv_batch(pentadsolver_handle_t handle, const double *ds,
                             const double *dl, const double *d,
                             const double *du, const double *dw, double *x,
                             const int *t_dims, int t_ndim, int t_solvedim,
                             void *t_buffer);

void pentadsolver_gpsv_batch(pentadsolver_handle_t handle, const float *ds,
                             const float *dl, const float *d, const float *du,
                             const float *dw, float *x, const int *t_dims,
                             int t_ndim, int t_solvedim, void *t_buffer);
#endif

#endif /* ifndef PENTADSOLVER_HPP_INCLUDED */
