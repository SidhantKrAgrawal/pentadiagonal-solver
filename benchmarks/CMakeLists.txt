option(BUILD_BENCHMARKS "Build the micro-benchmarks (Google Benchmark)" ON)

if(BUILD_BENCHMARKS)
  # Use an installed Google Benchmark if present; otherwise fetch it at
  # configure time (needs internet, see README). Replaces a hard-coded
  # personal install path. Disable the whole target with -DBUILD_BENCHMARKS=OFF.
  find_package(benchmark QUIET)
  if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found locally; fetching v1.8.3 from GitHub.")
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      benchmark
      URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz)
    FetchContent_MakeAvailable(benchmark)
  endif()
  if(BUILD_FOR_CPU)
    add_subdirectory(cpu)
  endif(BUILD_FOR_CPU)

  if(BUILD_FOR_CUDA)
    add_subdirectory(cuda)
  endif(BUILD_FOR_CUDA)
endif(BUILD_BENCHMARKS)
