#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_NOSTDOUT
#include <mpi.h>            // for MPI_COMM_WORLD, MPI_Abort, MPI_Barrier
#include <cstdio>           // for printf
#include <catch2/catch.hpp> // for Session
#include <cstdlib>          // for srand
#include <iostream>         // for operator<<, basic_ostream, basic_ostream...
#include <string>           // for operator<<, char_traits, basic_string
#include "catch_mpi_outputs.hpp"

// New cerr/cout/clog to separate the outputs of the different processes
namespace catch_mpi_outputs {
// NOLINTNEXTLINE
std::stringstream cout, cerr, clog;
} // namespace catch_mpi_outputs

int main(int argc, char *argv[]) {
  auto rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf("Error starting MPI program. Terminating.\n"); // NOLINT
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  // For the debug prints
  int rank     = 0;
  int num_proc = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::srand(rank);

  int result = Catch::Session().run(argc, argv);

  // Print the outputs
  for (int i = 0; i < num_proc; ++i) {
    if (i == rank) {
      if (!catch_mpi_outputs::cout.str().empty()) {
        std::cout << "##########################\n"
                  << "Rank " << i << " stdout:\n"
                  << "##########################\n"
                  << catch_mpi_outputs::cout.str();
      }
      if (!catch_mpi_outputs::cerr.str().empty()) {
        std::cerr << "##########################\n"
                  << "Rank " << i << " stderr:\n"
                  << "##########################\n"
                  << catch_mpi_outputs::cerr.str();
      }
      if (!catch_mpi_outputs::clog.str().empty()) {
        std::clog << "##########################\n"
                  << "Rank " << i << " stdlog:\n"
                  << "##########################\n"
                  << catch_mpi_outputs::clog.str();
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return result;
}
