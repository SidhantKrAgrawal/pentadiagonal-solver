cmake_minimum_required(VERSION 3.18...3.23)

project(app LANGUAGES CXX)

option(BUILD_FOR_CPU "Build library for CPU architecture." ON)
option(BUILD_FOR_CUDA "Build library for CUDA architecture." OFF)
option(BUILD_FOR_SP "Build library for SP." ON)
option(BUILD_FOR_MPI "Build distributer library." OFF)
option(BUILD_FOR_SN  "Build library for a single node combnied with the above architectures." OFF)
option(USE_PROFILING "Build timing library for to measure timings." OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  option(PRODUCE_OPT_REPORT "Print intel optimisation reports" OFF)

  # Set compiler flags/options for Intel compiler
  set(FLAGS_INTEL_DEFAULT "-xHost -m64 -parallel -qno-offload")
  if (PRODUCE_OPT_REPORT)
    set(FLAGS_INTEL_DEFAULT "${FLAGS_INTEL_DEFAULT} -qopt-report=2 -qopt-report-phase=vec -qopt-report-file=stdout")
  endif (PRODUCE_OPT_REPORT)

  # cmake-format: off
  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}     ${FLAGS_INTEL_DEFAULT}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  # cmake-format: on
else (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  # Set compiler flags/options for GCC compiler
  set(FLAGS_GCC_DEFAULT "-march=core-avx2") # -march=core-avx2
  set(FLAGS_GCC_RELEASE "-O3 -fstrict-aliasing -finline-functions") # -ffast-math -fargument-noalias -fargument-
                                                                      # noalias-global

  # cmake-format: off
  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}         ${FLAGS_GCC_DEFAULT} -std=c++17")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -O0")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${FLAGS_GCC_RELEASE}")

  # cmake-format: on
endif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")

set(FPPREC
    1
    CACHE STRING "0 - single precision build; 1 - double precision build")
add_definitions(-DFPPREC=${FPPREC})

# This variable is set by project() in CMake 3.21+
string(
  COMPARE EQUAL
          "${CMAKE_SOURCE_DIR}"
          "${PROJECT_SOURCE_DIR}"
          PROJECT_IS_TOP_LEVEL)
if (BUILD_FOR_CPU)
  find_package(OpenMP REQUIRED)
  if (BUILD_FOR_SN)
    add_executable(app_cpu 2D_cavity_cpu.cpp)
    target_compile_definitions(app_cpu PRIVATE -DFPPREC=1)
    target_include_directories(app_cpu PRIVATE ${_project_options_SOURCE_DIR}/include)
    target_link_libraries(app_cpu PRIVATE  pentadsolver::project_warnings
    pentadsolver::project_options pentadsolver::cpu OpenMP::OpenMP_CXX)

    add_executable(app_cpu_orig ${PROJECT_SOURCE_DIR}/2D_cavity_cpu_orig.cpp)
    target_compile_definitions(app_cpu_orig PRIVATE -DFPPREC=1)
    target_include_directories(app_cpu_orig PRIVATE ${_project_options_SOURCE_DIR}/include)
    target_link_libraries(app_cpu_orig PRIVATE  pentadsolver::project_warnings
    pentadsolver::project_options pentadsolver::pentad_orig_cpu OpenMP::OpenMP_CXX)

    # 3-D ADI CPU app — methodology mirror of app_cuda for fair CPU vs GPU
    # end-to-end comparison (thread count set at runtime via OMP_NUM_THREADS).
    add_executable(adi_cpu adi_cpu.cpp)
    target_compile_definitions(adi_cpu PRIVATE -DFPPREC=1)
    target_include_directories(adi_cpu PRIVATE ${_project_options_SOURCE_DIR}/include)
    target_link_libraries(adi_cpu PRIVATE  pentadsolver::project_warnings
    pentadsolver::project_options pentadsolver::cpu OpenMP::OpenMP_CXX)

    # Independent residual-based correctness check at arbitrary N (used for
    # sizes the fixed test suite doesn't cover, e.g. scale-up experiments).
    add_executable(verify_scale_cpu verify_scale_cpu.cpp)
    target_compile_definitions(verify_scale_cpu PRIVATE -DFPPREC=1)
    target_include_directories(verify_scale_cpu PRIVATE ${_project_options_SOURCE_DIR}/include)
    target_link_libraries(verify_scale_cpu PRIVATE  pentadsolver::project_warnings
    pentadsolver::project_options pentadsolver::cpu OpenMP::OpenMP_CXX)

  endif (BUILD_FOR_SN)
  if (BUILD_FOR_MPI)
    find_package(MPI REQUIRED)
    # 3-D ADI MPI CPU app — z-slab decomposition; methodology mirror of
    # adi_cpu for MPI scaling studies (hybrid via OMP_NUM_THREADS per rank).
    add_executable(adi_mpi_cpu adi_mpi_cpu.cpp)
    target_compile_definitions(adi_mpi_cpu PRIVATE -DFPPREC=1)
    target_include_directories(adi_mpi_cpu PRIVATE ${_project_options_SOURCE_DIR}/include)
    target_link_libraries(adi_mpi_cpu PRIVATE  pentadsolver::project_warnings
    pentadsolver::project_options pentadsolver::mpi_cpu OpenMP::OpenMP_CXX MPI::MPI_CXX)
  endif (BUILD_FOR_MPI)
endif (BUILD_FOR_CPU)
if (BUILD_FOR_CUDA)
  if (BUILD_FOR_SN)
    add_executable(app_cuda adi_cuda.cu)
    target_compile_definitions(app_cuda PRIVATE -DFPPREC=1)
    target_link_libraries(app_cuda PRIVATE
      pentadsolver::project_warnings
      pentadsolver::project_options
      pentadsolver::cuda)
    target_link_cuda(app_cuda)

    add_executable(verify_scale_cuda verify_scale_cuda.cu)
    target_compile_definitions(verify_scale_cuda PRIVATE -DFPPREC=1)
    target_include_directories(verify_scale_cuda PRIVATE ${_project_options_SOURCE_DIR}/include)
    target_link_libraries(verify_scale_cuda PRIVATE
      pentadsolver::project_warnings
      pentadsolver::project_options
      pentadsolver::cuda)
    target_link_cuda(verify_scale_cuda)

    add_executable(verify_shared_fact_cuda verify_shared_fact_cuda.cu)
    target_compile_definitions(verify_shared_fact_cuda PRIVATE -DFPPREC=1)
    target_include_directories(verify_shared_fact_cuda PRIVATE ${_project_options_SOURCE_DIR}/include)
    target_link_libraries(verify_shared_fact_cuda PRIVATE
      pentadsolver::project_warnings
      pentadsolver::project_options
      pentadsolver::cuda)
    target_link_cuda(verify_shared_fact_cuda)

    add_executable(verify_strided_cuda verify_strided_cuda.cu)
    target_compile_definitions(verify_strided_cuda PRIVATE -DFPPREC=1)
    target_include_directories(verify_strided_cuda PRIVATE ${_project_options_SOURCE_DIR}/include)
    target_link_libraries(verify_strided_cuda PRIVATE
      pentadsolver::project_warnings
      pentadsolver::project_options
      pentadsolver::cuda)
    target_link_cuda(verify_strided_cuda)

    add_executable(fp32_accuracy_cuda fp32_accuracy_cuda.cu)
    target_compile_definitions(fp32_accuracy_cuda PRIVATE -DFPPREC=1)
    target_include_directories(fp32_accuracy_cuda PRIVATE ${_project_options_SOURCE_DIR}/include)
    target_link_libraries(fp32_accuracy_cuda PRIVATE
      pentadsolver::project_warnings
      pentadsolver::project_options
      pentadsolver::cuda)
    target_link_cuda(fp32_accuracy_cuda)

    # Counter-free roofline probe.  Measures this GPU's FP64 issue rate and the
    # achievable memory floor for the solver's 7-array traffic, using only
    # timing — no ncu, no nsys metrics, no privileged counters.
    add_executable(roofline_probe_cuda roofline_probe_cuda.cu)
    target_include_directories(roofline_probe_cuda PRIVATE ${_project_options_SOURCE_DIR}/include)
    target_link_libraries(roofline_probe_cuda PRIVATE
      pentadsolver::project_options)
    target_link_cuda(roofline_probe_cuda)
  endif (BUILD_FOR_SN)
endif (BUILD_FOR_CUDA)

if(NOT TARGET pentadsolver::project_options)
    message(FATAL_ERROR "Requiered config package not found!")
    return() # be strictly paranoid for Template Janitor github action! CK
endif()


