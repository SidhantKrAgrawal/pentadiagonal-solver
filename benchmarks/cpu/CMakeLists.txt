
add_executable(my_bm my_bm.cpp)


target_link_libraries(
  my_bm
  PRIVATE OpenMP::OpenMP_CXX 
          pentadsolver::project_warnings
          pentadsolver::project_options
          pentadsolver::cpu
          benchmark::benchmark)
if(USE_PROFILING)
  target_link_libraries(
  my_bm PRIVATE profiling)
endif(USE_PROFILING)


