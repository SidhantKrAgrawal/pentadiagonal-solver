#include <catch2/catch.hpp>        // for INTERNAL_CATCH_NOINTERNAL_CATCH_DEF
#include <filesystem>              // for path
#include <vector>                  // for allocator, vector
#include "catch_utils.hpp"         // for require_allclose
#include "pentadsolver.hpp"        // for pentadsolver_create, pentadsolver...
#include "pentadsolver_handle.hpp" // for pentadsolver_handle_t
#include "util/mesh.hpp"           // for Mesh
namespace mpl_ {
struct na;
} // namespace mpl_

template <typename Float>
void test_from_file(const std::filesystem::path &file_name) {
  Mesh<Float> mesh(file_name);
  pentadsolver_handle_t handle{};
  pentadsolver_gpsv_batch(handle,             // context
                          mesh.ds().data(),   // ds
                          mesh.dl().data(),   // dl
                          mesh.d().data(),    // d
                          mesh.du().data(),   // du
                          mesh.dw().data(),   // dw
                          mesh.x().data(),    // x
                          mesh.dims().data(), // t_dims
                          mesh.dims().size(), // t_dims
                          mesh.solve_dim(),   // t_solvedim
                          nullptr);           // t_buffer

  require_allclose(mesh.u(), mesh.x());
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

TEMPLATE_TEST_CASE("y_solve: batch large", "[large]",double, float) { // NOLINT
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
