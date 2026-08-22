#include <benchmark/benchmark.h>
#include "util/device_mesh.hpp"
#include "pentadsolver.hpp"

// Ranges are {system size, ndims, solve_dim}.
template <typename Float> static void BM_PentadSolver(benchmark::State &state) {
  DeviceMesh<Float> mesh(
      state.range(2),
      std::vector<int>(state.range(1), static_cast<int>(state.range(0))));
  Float *x_d = nullptr;
  cudaMalloc((void **)&x_d, sizeof(Float) * mesh.x().size());
  cudaMemcpy(x_d, mesh.x_d(), sizeof(Float) * mesh.x().size(),
             cudaMemcpyDeviceToDevice);
  pentadsolver_handle_t handle{};
  pentadsolver_create(&handle, nullptr, 0, nullptr);

  // Allocated outside the timed loop so allocation cost is not measured.
  size_t buf_bytes = pentadsolver_gpsv_batch_buffer_size_ext(
      handle,
      mesh.ds_d(), mesh.dl_d(), mesh.d_d(), mesh.du_d(), mesh.dw_d(),
      mesh.x_d(), mesh.dims().data(), static_cast<int>(mesh.dims().size()),
      mesh.solve_dim());
  void *t_buf = nullptr;
  if (buf_bytes > 0) { cudaMalloc(&t_buf, buf_bytes); }

  for (auto _ : state) {
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
    // Restore the RHS outside the timed region so each iteration solves the
    // same system.
    state.PauseTiming();
    cudaMemcpy(mesh.x_d(), x_d, sizeof(Float) * mesh.x().size(),
               cudaMemcpyDeviceToDevice);
    state.ResumeTiming();
  }

  if (t_buf) { cudaFree(t_buf); }
  cudaFree(x_d);
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
