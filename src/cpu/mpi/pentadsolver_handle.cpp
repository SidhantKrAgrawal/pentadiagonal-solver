#include "pentadsolver_handle.hpp"
#include <mpi.h>
#include <vector>
#include <algorithm>

// ----------------------------------------------------------------------------
// Implementations
// ----------------------------------------------------------------------------

pentadsolver_handle_implementation::pentadsolver_handle_implementation(
    MPI_Comm cartesian_communicator, int t_num_dims, const int *t_num_mpi_procs)
    : communicators(t_num_dims), cart_groups(t_num_dims),
      neighbours_groups(t_num_dims), num_mpi_procs(t_num_mpi_procs),
      mpi_coords(t_num_dims) {
  int cart_rank = 0;
  MPI_Comm_rank(cartesian_communicator, &cart_rank);
  MPI_Cart_coords(cartesian_communicator, cart_rank, t_num_dims,
                  this->mpi_coords.data());
  for (int equation_dim = 0; equation_dim < t_num_dims; ++equation_dim) {
    std::vector<int> neighbours = {cart_rank};
    int mpi_coord               = this->mpi_coords[equation_dim];
    // Collect the processes in the same row/column
    for (int i = 1;
         i <= std::max(num_mpi_procs[equation_dim] - mpi_coord - 1, mpi_coord);
         ++i) {
      int prev = 0;
      int next = 0;
      MPI_Cart_shift(cartesian_communicator, equation_dim, i, &prev, &next);
      if (i <= mpi_coord) {
        neighbours.push_back(prev);
      }
      if (i + mpi_coord < num_mpi_procs[equation_dim]) {
        neighbours.push_back(next);
      }
    }

    // This is needed, otherwise the communications hang
    std::sort(neighbours.begin(), neighbours.end());

    // Create new communicator for neighbours
    // MPI_Group cart_group;
    MPI_Comm_group(cartesian_communicator, &this->cart_groups[equation_dim]);
    // MPI_Group neighbours_group;
    MPI_Group_incl(this->cart_groups[equation_dim],
                   static_cast<int>(neighbours.size()), neighbours.data(),
                   &this->neighbours_groups[equation_dim]);
    MPI_Comm_create(cartesian_communicator,
                    this->neighbours_groups[equation_dim],
                    &this->communicators[equation_dim]);
  }
}

pentadsolver_handle_implementation::~pentadsolver_handle_implementation() {
  for (unsigned int equation_dim = 0; equation_dim < this->communicators.size();
       ++equation_dim) {
    MPI_Group_free(&this->cart_groups[equation_dim]);
    MPI_Group_free(&this->neighbours_groups[equation_dim]);
    MPI_Comm_free(&this->communicators[equation_dim]);
  }
}
