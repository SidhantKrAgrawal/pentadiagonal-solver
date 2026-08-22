cmake_minimum_required(VERSION 3.18...3.23)

# ---------------------------------------------------------------------------
# CUDA target architecture.  Default to "native" (auto-detect the build
# machine's GPU, CMake >= 3.24) so a fresh checkout builds without extra
# flags on a GPU node.  Override for cross-compilation or when building on a
# login node that has no GPU, e.g.:
#     cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86    # RTX 3050 / Ampere
#     cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80    # A100
#     cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90    # H100
# The provided scripts/build.sh auto-detects this via nvidia-smi.
# ---------------------------------------------------------------------------
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES
      "native"
      CACHE STRING "CUDA compute capability, e.g. 86 (RTX 3050), 80 (A100), 90 (H100); 'native' auto-detects the build GPU")
endif ()

# Not ideal to use this global variable, but necessary to make sure
# that tooling and projects use the same version
set(CMAKE_CXX_STANDARD 17) # TODO might change

#set the below 2 flags to avoid error in GCC
set(CMAKE_C_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel." FORCE)
endif (NOT CMAKE_BUILD_TYPE)
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  option(PRODUCE_OPT_REPORT "Print intel optimisation reports" ON)

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

# strongly encouraged to enable this globally to avoid conflicts between
# -Wpedantic being enabled and -std=c++20 and -std=gnu++20 for example
# when compiling with PCH enabled
set(CMAKE_CXX_EXTENSIONS OFF)

# Developer mode enables sanitizers, static analysis (cppcheck/clang-tidy)
# and treats all warnings as errors.  It is OFF by default here so that a
# plain build succeeds across the wide range of compiler/toolchain versions
# found on HPC clusters (those extra tools are frequently missing or a
# different version, which otherwise breaks the build).  Contributors can
# re-enable it with -DENABLE_DEVELOPER_MODE=ON.
set(ENABLE_DEVELOPER_MODE
    FALSE
    CACHE BOOL "Enable 'developer mode' (sanitizers, static analysis, warnings-as-errors)")

# Change this to false if you want to disable warnings_as_errors in developer mode
set(OPT_WARNINGS_AS_ERRORS_DEVELOPER_DEFAULT TRUE)

# Add project_options v0.20.0
# https://github.com/cpp-best-practices/project_options
include(FetchContent)
FetchContent_Declare(_project_options URL https://github.com/aminya/project_options/archive/refs/tags/v0.20.0.zip)
FetchContent_MakeAvailable(_project_options)
include(${_project_options_SOURCE_DIR}/Index.cmake)

# Set the project name and language
project(
  pentadsolver
  VERSION 0.0.1
  DESCRIPTION ""
  LANGUAGES CXX)

# This variable is set by project() in CMake 3.21+
string(
  COMPARE EQUAL
          "${CMAKE_SOURCE_DIR}"
          "${PROJECT_SOURCE_DIR}"
          PROJECT_IS_TOP_LEVEL)

get_property(BUILDING_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(BUILDING_MULTI_CONFIG)
  if(NOT CMAKE_BUILD_TYPE)
    # Make sure that all supported configuration types have their
    # associated conan packages available. You can reduce this
    # list to only the configuration types you use, but only if one
    # is not forced-set on the command line for VS
    message(TRACE "Setting up multi-config build types")
    set(CMAKE_CONFIGURATION_TYPES
        Debug
        Release
        RelWithDebInfo
        MinSizeRel
        CACHE STRING "Enabled build types" FORCE)
  else()
    message(TRACE "User chose a specific build type, so we are using that")
    set(CMAKE_CONFIGURATION_TYPES
        ${CMAKE_BUILD_TYPE}
        CACHE STRING "Enabled build types" FORCE)
  endif()
endif()

include(${_project_options_SOURCE_DIR}/src/DynamicProjectOptions.cmake)

# defaulted_project_options sets recommended defaults and provides user and developer
# modes and full GUI support for choosing options at configure time

# for more flexibility, look into project_options() macro

# Any default can be overridden
# set(<feature_name>_DEFAULT <value>) - set default for both user and developer modes
# set(<feature_name>_DEVELOPER_DEFAULT <value>) - set default for developer mode
# set(<feature_name>_USER_DEFAULT <value>) - set default for user mode

# Initialize project_options variable related to this project
# This overwrites `project_options` and sets `project_warnings`
# uncomment the options to enable them:
dynamic_project_options(
     #ENABLE_COVERAGE
     #ENABLE_CLANG_TIDY
  # ENABLE_CONAN
  # CONAN_OPTIONS    # Extra options to pass to conan
  # MSVC_WARNINGS    # Override the defaults for the MSVC warnings
  # CLANG_WARNINGS   # Override the defaults for the CLANG warnings
  # GCC_WARNINGS     # Override the defaults for the GCC warnings
  CPPCHECK_OPTIONS
  --enable=style,performance,warning,portability
  --inline-suppr
   #We cannot act on a bug/missing feature of cppcheck
  --suppress=cppcheckError
  --suppress=internalAstError
   #if a file does not have an internalAstError, we get an unmatchedSuppression error
  --suppress=unmatchedSuppression
 --suppress=passedByValue
  --suppress=syntaxError
  --inconclusive
   )
target_compile_features(project_options INTERFACE cxx_std_${CMAKE_CXX_STANDARD})
add_library(${PROJECT_NAME}::project_options INTERFACE IMPORTED)
add_library(${PROJECT_NAME}::project_warnings INTERFACE IMPORTED)


find_package(OpenMP REQUIRED)
# Adding the src:
add_subdirectory(src)

add_subdirectory(benchmarks)
add_subdirectory(apps)

# Adding the tests:
option(ENABLE_TESTING "Enable the tests" ${PROJECT_IS_TOP_LEVEL})
if(ENABLE_TESTING)
  enable_testing()
  message(AUTHOR_WARNING "Building Tests.")
  add_subdirectory(test)
endif()
