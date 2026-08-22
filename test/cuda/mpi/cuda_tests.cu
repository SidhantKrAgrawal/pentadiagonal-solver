#include <mpi.h>                   // for MPI_COMM_WORLD, MPI_Comm_rank
#include <cstddef>                 // for size_t
#include <catch2/catch.hpp>        // for INTERNAL_CATCH_NOINTERNAL_CATCH_DEF
#include <filesystem>              // for path
#include <vector>                  // for allocator, vector
#include "catch_utils.hpp"         // for require_allclose
#include "util/device_mesh.hpp"    // for DeviceMesh
#include "pentadsolver_cuda.hpp"   // for pentadsolver_gpsv_batch
#include "pentadsolver_handle.hpp" // for pentadsolver_handle_implementation
#include "catch_mpi_outputs.hpp"

template <typename Float>
void test_solver_from_file(const std::filesystem::path &file_name) {
  // The dimension of the MPI decomposition is the same as solve_dim
  Mesh<Float> mesh_h(file_name);

  int num_proc = 0;
  int rank     = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(mesh_h.dims().size());
  std::vector<int> periods(mesh_h.dims().size(), 0);
  mpi_dims[mesh_h.solve_dim()] = num_proc;
  MPI_Dims_create(num_proc, mesh_h.dims().size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm{};
  MPI_Cart_create(MPI_COMM_WORLD, mesh_h.dims().size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  pentadsolver_handle_t handle{};
  pentadsolver_create(&handle, &cart_comm, mesh_h.dims().size(),
                      mpi_dims.data());

  // The size of the local domain.
  std::vector<int> local_sizes(mesh_h.dims().size());
  // The starting indices of the local domain in each dimension.
  std::vector<int> domain_offsets(mesh_h.dims().size());
  // The strides in the mesh_h for each dimension.
  std::vector<int> global_strides(mesh_h.dims().size());
  int domain_size = 1;
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = mesh_h.dims()[i];
    domain_offsets[i]    = handle->mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i]       = handle->mpi_coords[i] == mpi_dims[i] - 1
                               ? global_dim - domain_offsets[i]
                               : global_dim / mpi_dims[i];
    global_strides[i] =
        i == 0 ? 1 : global_strides[i - 1] * mesh_h.dims()[i - 1];
    domain_size *= local_sizes[i];
  }

  // Simulate distributed environment: only load our data
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

  copy_strided(mesh_h.ds(), ds, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh_h.dl(), dl, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh_h.d(), d, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh_h.du(), du, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh_h.dw(), dw, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh_h.x(), x, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh_h.u(), u, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);

  auto init_arr = [](Float **arr_d, const std::vector<Float> &arr_h) {
    cudaMalloc((void **)arr_d, sizeof(Float) * arr_h.size());
    cudaMemcpy(*arr_d, arr_h.data(), sizeof(Float) * arr_h.size(),
               cudaMemcpyHostToDevice);
  };
  Float *ds_d = nullptr;
  Float *dl_d = nullptr;
  Float *d_d  = nullptr;
  Float *du_d = nullptr;
  Float *dw_d = nullptr;
  Float *x_d  = nullptr;
  init_arr(&ds_d, ds);
  init_arr(&dl_d, dl);
  init_arr(&d_d, d);
  init_arr(&du_d, du);
  init_arr(&dw_d, dw);
  init_arr(&x_d, x);

  size_t workspace_size_h = 0;
  size_t workspace_size_d = 0;
  pentadsolver_gpsv_batch_buffer_size_ext(handle,               // context
                                          ds_d,                 // ds
                                          dl_d,                 // dl
                                          d_d,                  // d
                                          du_d,                 // du
                                          dw_d,                 // dw
                                          x_d,                  // x
                                          local_sizes.data(),   // t_dims
                                          mesh_h.dims().size(), // t_ndims
                                          mesh_h.solve_dim(),   // t_solvedim
                                          &workspace_size_h, &workspace_size_d);
  std::vector<char> buffer(workspace_size_h, 0);
  void *workspace_d = nullptr;
  cudaMalloc(&workspace_d, workspace_size_d);

  // Solve the equations
  pentadsolver_gpsv_batch(handle,               // context
                          ds_d,                 // ds
                          dl_d,                 // dl
                          d_d,                  // d
                          du_d,                 // du
                          dw_d,                 // dw
                          x_d,                  // x
                          local_sizes.data(),   // t_dims
                          mesh_h.dims().size(), // t_ndims
                          mesh_h.solve_dim(),   // t_solvedim
                          buffer.data(),        // t_buffer_h
                          workspace_d);         // t_buffer_d

  cudaMemcpy(x.data(), x_d, x.size() * sizeof(Float), cudaMemcpyDeviceToHost);
  require_allclose(u, x);
  pentadsolver_destroy(&handle);
}

TEMPLATE_TEST_CASE("x_solve: batch small", "[small]", double, float) { // NOLINT
  // SECTION("ndims: 1") {
  //   test_solver_from_file<TestType>("files/one_dim_small_solve0");
  // }
  SECTION("ndims: 2") {
    test_solver_from_file<TestType>("files/two_dim_small_solve0");
  }
}

TEMPLATE_TEST_CASE("y_solve: batch small", "[small]", double, float) { // NOLINT
  SECTION("ndims: 2") {
    test_solver_from_file<TestType>("files/two_dim_small_solve1");
  }
}

TEMPLATE_TEST_CASE("x_solve: batch large", "[large]", double, float) { // NOLINT
  SECTION("ndims: 1") {
    test_solver_from_file<TestType>("files/one_dim_large_solve0");
  }
  SECTION("ndims: 2") {
    test_solver_from_file<TestType>("files/two_dim_large_solve0");
  }
  SECTION("ndims: 3") {
    test_solver_from_file<TestType>("files/three_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE("y_solve: batch large", "[large]", double, float) { // NOLINT
  SECTION("ndims: 2") {
    test_solver_from_file<TestType>("files/two_dim_large_solve1");
  }
  SECTION("ndims: 3") {
    test_solver_from_file<TestType>("files/three_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE("z_solve: batch large", "[large]", double, float) { // NOLINT
  SECTION("ndims: 3") {
    test_solver_from_file<TestType>("files/three_dim_large_solve2");
  }
}
