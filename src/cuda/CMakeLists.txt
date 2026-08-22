add_library(cuda ./singlenode/pentad_cuda.cu)
add_library(${PROJECT_NAME}::cuda ALIAS cuda)

target_link_libraries(cuda PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME}::project_options
                                   ${PROJECT_NAME}::project_warnings)
target_link_cuda(cuda)

# TODO CUDA_ARCH??

target_include_directories(cuda PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                       $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)

if(BUILD_FOR_MPI)
  find_package(MPI REQUIRED)

  add_library(cuda_mpi_handle ../cpu/mpi/pentadsolver_handle.cpp)
  add_library(${PROJECT_NAME}::cuda_mpi_handle ALIAS cuda_mpi_handle)
  target_link_libraries(
    cuda_mpi_handle
    PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME}::project_options ${PROJECT_NAME}::project_warnings
    PUBLIC MPI::MPI_CXX)

  target_compile_definitions(cuda_mpi_handle PUBLIC PENTADSOLVER_MPI)

  target_include_directories(cuda_mpi_handle PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                    $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)

  add_library(mpi_cuda ./mpi/pentad_cuda.cu)
  add_library(${PROJECT_NAME}::mpi_cuda ALIAS mpi_cuda)

  target_link_libraries(
    mpi_cuda
    PRIVATE OpenMP::OpenMP_CXX
            ${PROJECT_NAME}::project_options
            ${PROJECT_NAME}::project_warnings
            ${PROJECT_NAME}::cuda_mpi_handle
    PUBLIC MPI::MPI_CXX)

  target_compile_definitions(mpi_cuda PUBLIC PENTADSOLVER_MPI)

  target_include_directories(mpi_cuda PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                             $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)
endif(BUILD_FOR_MPI)
