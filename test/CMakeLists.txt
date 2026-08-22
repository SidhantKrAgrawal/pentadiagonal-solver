cmake_minimum_required(VERSION 3.18...3.23)

project(PentadSolverPackageTests LANGUAGES CXX)

# ---- Test as standalone project the exported config package ----
# This variable is set by project() in CMake 3.21+
string(
  COMPARE EQUAL
          "${CMAKE_SOURCE_DIR}"
          "${PROJECT_SOURCE_DIR}"
          PROJECT_IS_TOP_LEVEL)
if(PROJECT_IS_TOP_LEVEL OR TEST_INSTALLED_VERSION)
  enable_testing()

  find_package(pentadsolver CONFIG REQUIRED) # for project_options, ...

  if(NOT TARGET pentadsolver::project_options)
    message(FATAL_ERROR "Requiered config package not found!")
    return() # be strictly paranoid for Template Janitor github action! CK
  endif()
endif()

# ---- Dependencies: Catch2 (v2) ----
# Use an installed Catch2 v2 if present; otherwise fetch it from source at
# configure time (needs internet on the configuring machine, see README).
# This replaces a previously hard-coded personal install path that only worked
# on the original author's account.
find_package(Catch2 2 QUIET)
if(NOT Catch2_FOUND)
  message(STATUS "Catch2 not found locally; fetching v2.13.10 from GitHub.")
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v2.13.10.tar.gz)
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
endif()
include(Catch)
add_library(catch_main OBJECT catch_main.cpp)
target_link_libraries(catch_main PUBLIC Catch2::Catch2)
target_link_libraries(catch_main PRIVATE pentadsolver::project_options)

if(BUILD_FOR_CPU)
  add_subdirectory(cpu)
endif(BUILD_FOR_CPU)

if(BUILD_FOR_CUDA)
  add_subdirectory(cuda)
endif(BUILD_FOR_CUDA)
