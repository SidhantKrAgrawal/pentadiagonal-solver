#include <benchmark/benchmark.h>
#include "util/mesh.hpp"
#include "pentadsolver.hpp"
#include <iostream>

// Ranges are {system size, ndims, solve_dim}.
template <typename Float> static void BM_PentadSolver(benchmark::State &state) {
  Mesh<Float> mesh(
      state.range(2),
      std::vector<int>(state.range(1), static_cast<int>(state.range(0))));
  std::vector<Float> x(mesh.x());
  pentadsolver_handle_t handle{};
  for (auto _ : state) {
    pentadsolver_gpsv_batch(handle,             // context
                            mesh.ds().data(),   // ds
                            mesh.dl().data(),   // dl
                            mesh.d().data(),    // d
                            mesh.du().data(),   // du
                            mesh.dw().data(),   // dw
                            mesh.x().data(),    // x
                            mesh.dims().data(), // t_dims
                            mesh.dims().size(), // t_ndim
                            mesh.solve_dim(),   // t_solvedim
                            nullptr);           // t_buffer
    // Restore the RHS outside the timed region so each iteration solves the
    // same system.
    state.PauseTiming();
    mesh.x() = x;
    state.ResumeTiming();
  }
}

constexpr int range_min = 32;
constexpr int range_max = 2U << 7U;

// NOLINTBEGIN(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-owning-memory)
BENCHMARK_TEMPLATE(BM_PentadSolver, float)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {1, 1}, {0, 0}});
BENCHMARK_TEMPLATE(BM_PentadSolver, double)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {1, 1}, {0, 0}});
BENCHMARK_TEMPLATE(BM_PentadSolver, float)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {2, 2}, {0, 1}});
BENCHMARK_TEMPLATE(BM_PentadSolver, double)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {2, 2}, {0, 1}});
BENCHMARK_TEMPLATE(BM_PentadSolver, float)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {3, 3}, {0, 2}});
BENCHMARK_TEMPLATE(BM_PentadSolver, double)
    ->RangeMultiplier(2)
    ->Ranges({{range_min, range_max}, {3, 3}, {0, 2}});
// NOLINTEND(cert-err58-cpp,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-owning-memory)

BENCHMARK_MAIN();
