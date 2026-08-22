#ifndef PENTADSOLVER_HANDLE_HPP_INCLUDED
#define PENTADSOLVER_HANDLE_HPP_INCLUDED

#ifndef PENTADSOLVER_MPI
using pentadsolver_handle_t = void *;
#else
#include <mpi.h>
#include <vector>

struct pentadsolver_handle_implementation {
  // This will be an array with a communicator for each dimension. Separate
  // communicator that includes every node calculating the same set of equations
  // as the current node for each dimension.
  std::vector<MPI_Comm> communicators;
  std::vector<MPI_Group> cart_groups;
  std::vector<MPI_Group> neighbours_groups;

  // The number of MPI processes in each dimension. It is `num_dims` large. It
  // won't be owned.
  const int *num_mpi_procs;

  // The coordinates of the current MPI process in the cartesian mesh.
  std::vector<int> mpi_coords;

  // ----------------------------------------------------------------------------
  // Implementations
  // ----------------------------------------------------------------------------

  pentadsolver_handle_implementation(MPI_Comm cartesian_communicator,
                                     int t_num_dims,
                                     const int *t_num_mpi_procs);
  pentadsolver_handle_implementation(
      const pentadsolver_handle_implementation &) = delete;
  pentadsolver_handle_implementation &
  operator=(const pentadsolver_handle_implementation &) = delete;
  pentadsolver_handle_implementation(pentadsolver_handle_implementation &&) =
      delete;
  pentadsolver_handle_implementation &
  operator=(pentadsolver_handle_implementation &&) = delete;
  ~pentadsolver_handle_implementation();
};

using pentadsolver_handle_t = pentadsolver_handle_implementation *;

#endif /* ifndef PENTADSOLVER_MPI */
#endif /* ifndef PENTADSOLVER_HANDLE_HPP_INCLUDED */
