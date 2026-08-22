#include <mpi.h>                   // for MPI_COMM_WORLD, MPI_Comm_rank
#include <cstddef>                 // for size_t
#include <catch2/catch.hpp>        // for INTERNAL_CATCH_NOINTERNAL_CATCH_DEF
#include <filesystem>              // for path
#include <vector>                  // for vector, allocator
#include "catch_utils.hpp"         // for copy_strided, require_allclose
#include "pentadsolver.hpp"        // for pentadsolver_destroy, pentadsolve...
#include "pentadsolver_handle.hpp" // for pentadsolver_handle_implementation
#include "util/mesh.hpp"           // for Mesh
#include "catch_mpi_outputs.hpp"
namespace mpl_ {
struct na;
} // namespace mpl_

template <typename Float>
void test_solver_from_file(const std::filesystem::path &file_name) {
  // The dimension of the MPI decomposition is the same as solve_dim
  Mesh<Float> mesh(file_name);

  int num_proc = 0;
  int rank     = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(mesh.dims().size());
  std::vector<int> periods(mesh.dims().size(), 0);
  mpi_dims[mesh.solve_dim()] = num_proc;
  MPI_Dims_create(num_proc, mesh.dims().size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm{};
  MPI_Cart_create(MPI_COMM_WORLD, mesh.dims().size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  pentadsolver_handle_t handle{};
  pentadsolver_create(&handle, &cart_comm, mesh.dims().size(), mpi_dims.data());

  // The size of the local domain.
  std::vector<int> local_sizes(mesh.dims().size());
  // The starting indices of the local domain in each dimension.
  std::vector<int> domain_offsets(mesh.dims().size());
  // The strides in the mesh for each dimension.
  std::vector<int> global_strides(mesh.dims().size());
  int domain_size = 1;
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = mesh.dims()[i];
    domain_offsets[i]    = handle->mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i]       = handle->mpi_coords[i] == mpi_dims[i] - 1
                               ? global_dim - domain_offsets[i]
                               : global_dim / mpi_dims[i];
    global_strides[i] = i == 0 ? 1 : global_strides[i - 1] * mesh.dims()[i - 1];
    domain_size *= local_sizes[i];
  }

  // Simulate distributed environment: only load the local data
  std::vector<Float> ds;
  std::vector<Float> dl;
  std::vector<Float> d;
  std::vector<Float> du;
  std::vector<Float> dw;
  std::vector<Float> x;
  std::vector<Float> u;

  ds.reserve(domain_size);
  dl.reserve(domain_size);
  d.reserve(domain_size);
  du.reserve(domain_size);
  dw.reserve(domain_size);
  x.reserve(domain_size);
  u.reserve(domain_size);

  copy_strided(mesh.ds(), ds, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.dl(), dl, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.d(), d, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.du(), du, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.dw(), dw, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.x(), x, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.u(), u, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);

  size_t extent =
      pentadsolver_gpsv_batch_buffer_size_ext(handle,             // context
                                              ds.data(),          // ds
                                              dl.data(),          // dl
                                              d.data(),           // d
                                              du.data(),          // du
                                              dw.data(),          // dw
                                              x.data(),           // x
                                              local_sizes.data(), // t_dims
                                              mesh.dims().size(), // t_ndims
                                              mesh.solve_dim());  // t_solvedim
  std::vector<char> buffer(extent, 0);

  // Solve the equations
  pentadsolver_gpsv_batch(handle,             // context
                          ds.data(),          // ds
                          dl.data(),          // dl
                          d.data(),           // d
                          du.data(),          // du
                          dw.data(),          // dw
                          x.data(),           // x
                          local_sizes.data(), // t_dims
                          mesh.dims().size(), // t_ndims
                          mesh.solve_dim(),   // t_solvedim
                          buffer.data());     // t_buffer

  // Check result
  require_allclose(u, x);
  pentadsolver_destroy(&handle);
}

TEMPLATE_TEST_CASE("z_solve: batch large", "[large]", float) { // NOLINT
  SECTION("ndims: 3") {
    test_solver_from_file<TestType>("files/three_dim_large_solve2");
  }
}
