#include <catch2/catch.hpp>     // for INTERNAL_CATCH_NOINTERNAL_CATCH_DEF
#include <filesystem>           // for path
#include <vector>               // for allocator, vector
#include "catch_utils.hpp"      // for require_allclose
#include "util/device_mesh.hpp" // for DeviceMesh
#include "pentadsolver.hpp"     // for pentadsolver_gpsv_batch

template <typename Float>
void test_from_file(const std::filesystem::path &file_name) {
  DeviceMesh<Float> mesh(file_name);
  pentadsolver_handle_t handle{};
  pentadsolver_create(&handle, nullptr, 0, nullptr);

  size_t buf_bytes = pentadsolver_gpsv_batch_buffer_size_ext(
      handle, mesh.ds_d(), mesh.dl_d(), mesh.d_d(), mesh.du_d(), mesh.dw_d(),
      mesh.x_d(), mesh.dims().data(), static_cast<int>(mesh.dims().size()),
      mesh.solve_dim());
  void *t_buf = nullptr;
  if (buf_bytes > 0) { cudaMalloc(&t_buf, buf_bytes); }

  pentadsolver_gpsv_batch(handle,             // context
                          mesh.ds_d(),        // ds
                          mesh.dl_d(),        // dl
                          mesh.d_d(),         // d
                          mesh.du_d(),        // du
                          mesh.dw_d(),        // dw
                          mesh.x_d(),         // x
                          mesh.dims().data(), // t_dims
                          mesh.dims().size(), // t_ndim
                          mesh.solve_dim(),   // t_solvedim
                          t_buf);             // t_buffer

  cudaDeviceSynchronize();
  cudaMemcpy(mesh.x().data(), mesh.x_d(), mesh.x().size() * sizeof(Float),
             cudaMemcpyDeviceToHost);
  if (t_buf) { cudaFree(t_buf); }
  require_allclose(mesh.u(), mesh.x());
  pentadsolver_destroy(&handle);
}

TEMPLATE_TEST_CASE("x_solve: batch small", "[small]", double, float) { // NOLINT
  SECTION("ndims: 1") {
    test_from_file<TestType>("files/one_dim_small_solve0");
  }
  SECTION("ndims: 2") {
    test_from_file<TestType>("files/two_dim_small_solve0");
  }
}

TEMPLATE_TEST_CASE("y_solve: batch small", "[small]", double, float) { // NOLINT
  SECTION("ndims: 2") {
    test_from_file<TestType>("files/two_dim_small_solve1");
  }
}

TEMPLATE_TEST_CASE("x_solve: batch large", "[large]", double, float) { // NOLINT
  SECTION("ndims: 1") {
    test_from_file<TestType>("files/one_dim_large_solve0");
  }
  SECTION("ndims: 2") {
    test_from_file<TestType>("files/two_dim_large_solve0");
  }
  SECTION("ndims: 3") {
    test_from_file<TestType>("files/three_dim_large_solve0");
  }
}

TEMPLATE_TEST_CASE("y_solve: batch large", "[large]", double, float) { // NOLINT
  SECTION("ndims: 2") {
    test_from_file<TestType>("files/two_dim_large_solve1");
  }
  SECTION("ndims: 3") {
    test_from_file<TestType>("files/three_dim_large_solve1");
  }
}

TEMPLATE_TEST_CASE("z_solve: batch large", "[large]", double, float) { // NOLINT
  SECTION("ndims: 3") {
    test_from_file<TestType>("files/three_dim_large_solve2");
  }
}
