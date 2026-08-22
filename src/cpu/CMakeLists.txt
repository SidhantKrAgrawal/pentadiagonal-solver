
set(CPU_LIBS "OpenMP::OpenMP_CXX")
if (USE_PROFILING)
  set(CPU_LIBS "${CPU_LIBS};profiling")
  add_library(profiling ${BUILD_TYPE} ./singlenode/timing.cpp)
  target_include_directories(profiling PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                              $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)
  set_property(TARGET profiling PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(profiling PUBLIC -DPROFILING=1)
endif(USE_PROFILING)

add_library(pentad_global_options INTERFACE)
target_include_directories(pentad_global_options INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                         $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)
set(CPU_INTERFACE_LIBS "pentad_global_options")

if(BUILD_FOR_SN)
  add_library(pentadcpu_sp OBJECT ./singlenode/pentad_cpu.cpp)
  add_library(pentadcpu_dp OBJECT ./singlenode/pentad_cpu.cpp)
  add_library(cpu ${BUILD_TYPE} $<TARGET_OBJECTS:pentadcpu_sp> $<TARGET_OBJECTS:pentadcpu_dp>)
  add_library(${PROJECT_NAME}::cpu ALIAS cpu)

  target_link_libraries(pentadcpu_dp PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME}::project_options ${PROJECT_NAME}::project_warnings ${CPU_INTERFACE_LIBS} profiling )
  target_link_libraries(pentadcpu_sp PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME}::project_options ${PROJECT_NAME}::project_warnings ${CPU_INTERFACE_LIBS} profiling )
  target_link_libraries(cpu PRIVATE OpenMP::OpenMP_CXX PUBLIC ${CPU_INTERFACE_LIBS})

  set_property(TARGET pentadcpu_sp pentadcpu_dp PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(pentadcpu_sp PRIVATE -DFPPREC=0)
  target_compile_definitions(pentadcpu_dp PRIVATE -DFPPREC=1)

  add_library(pentad_orig_cpu_sp OBJECT ./singlenode/pentad_orig_cpu.cpp)
  add_library(pentad_orig_cpu_dp OBJECT ./singlenode/pentad_orig_cpu.cpp)
  add_library(pentad_orig_cpu ${BUILD_TYPE} $<TARGET_OBJECTS:pentad_orig_cpu_sp> $<TARGET_OBJECTS:pentad_orig_cpu_dp>)
  add_library(${PROJECT_NAME}::pentad_orig_cpu ALIAS pentad_orig_cpu)

  target_link_libraries(pentad_orig_cpu_dp PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME}::project_options ${PROJECT_NAME}::project_warnings ${CPU_INTERFACE_LIBS} profiling )
  target_link_libraries(pentad_orig_cpu_sp PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME}::project_options ${PROJECT_NAME}::project_warnings ${CPU_INTERFACE_LIBS} profiling )
  target_link_libraries(pentad_orig_cpu PRIVATE OpenMP::OpenMP_CXX PUBLIC ${CPU_INTERFACE_LIBS})

  set_property(TARGET pentad_orig_cpu_sp pentad_orig_cpu_dp PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(pentad_orig_cpu_sp PRIVATE -DFPPREC=0)
  target_compile_definitions(pentad_orig_cpu_dp PRIVATE -DFPPREC=1)
endif(BUILD_FOR_SN)


if(BUILD_FOR_MPI)
  find_package(MPI REQUIRED)

  add_library(mpi_handle ./mpi/pentadsolver_handle.cpp)
  add_library(${PROJECT_NAME}::mpi_handle ALIAS mpi_handle)
  target_link_libraries(
    mpi_handle
    PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME}::project_options ${PROJECT_NAME}::project_warnings
    PUBLIC MPI::MPI_CXX)

  target_compile_definitions(mpi_handle PUBLIC PENTADSOLVER_MPI)

  target_include_directories(mpi_handle PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                            $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)

  add_library(mpi_cpu ./mpi/pentad_cpu.cpp)
  add_library(${PROJECT_NAME}::mpi_cpu ALIAS mpi_cpu)

  target_link_libraries(
    mpi_cpu
    PRIVATE OpenMP::OpenMP_CXX
            ${PROJECT_NAME}::project_options
            ${PROJECT_NAME}::project_warnings
            ${PROJECT_NAME}::mpi_handle
    PUBLIC MPI::MPI_CXX)

  target_compile_definitions(mpi_cpu PUBLIC PENTADSOLVER_MPI)

  target_include_directories(mpi_cpu PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                            $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)
endif(BUILD_FOR_MPI)
