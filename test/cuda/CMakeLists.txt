add_executable(cuda_tests singlenode/cuda_tests.cu)
target_link_libraries(
  cuda_tests
  PRIVATE pentadsolver::project_warnings
          pentadsolver::project_options
          pentadsolver::cuda
          catch_main)

target_include_directories(cuda_tests PUBLIC ${PROJECT_SOURCE_DIR})
target_link_cuda(cuda_tests)

# automatically discover tests that are defined in catch based test files you can modify the unittests. Set TEST_PREFIX
# to whatever you want, or use different for different binaries
catch_discover_tests(
  cuda_tests
  TEST_PREFIX
  "unittests."
  REPORTER
  xml
  WORKING_DIRECTORY
  ${PROJECT_SOURCE_DIR}
  OUTPUT_DIR
  .OUTPUT_PREFIX
  "unittests."
  OUTPUT_SUFFIX.xml)

if(BUILD_FOR_MPI)
  find_package(MPI REQUIRED)

  add_executable(cuda_mpi_tests mpi/mpi_main.cpp mpi/cuda_tests.cu)
  target_link_libraries(
    cuda_mpi_tests
    PRIVATE pentadsolver::project_warnings
            pentadsolver::project_options
            pentadsolver::mpi_cuda
            Catch2::Catch2)

  target_include_directories(cuda_mpi_tests PUBLIC ${PROJECT_SOURCE_DIR})

  catch_discover_tests(
    cuda_mpi_tests
    TEST_PREFIX
    "unittests."
    REPORTER
    xml
    WORKING_DIRECTORY
    ${PROJECT_SOURCE_DIR}
    OUTPUT_DIR
    .OUTPUT_PREFIX
    "unittests."
    OUTPUT_SUFFIX.xml)

endif(BUILD_FOR_MPI)
