cmake_minimum_required (VERSION 3.4)

# Project name and version.
project (hpp)
set (hpp_VERSION "0.5.0")

# COMPILERS #
#############
set(CMAKE_CXX_STANDARD 11)

# MODULES #
###########

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Intra-project directories
# These need to be included early so that they'll supersede installed versions
# of this library that reside in standard system paths on systems where we're 
# actively developing it.
include_directories(include)

# FIND PACKAGES #
#################

# OpenMP
find_package(OpenMP REQUIRED)

# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

# LAPACK
find_package(LAPACK REQUIRED)

# BLAS
find_package(BLAS REQUIRED)

# Eigen
find_package(Eigen REQUIRED)
message("Eigen includes ${EIGEN_INCLUDE_DIRS}")
include_directories(${EIGEN_INCLUDE_DIRS})

# HDF5
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
if (NOT ${HDF5_IS_PARALLEL})
    message(FATAL_ERROR "Parallel HDF5 not found.")
endif()
include_directories(${HDF5_INCLUDE_DIRS})

# FFTW
find_package(FFTW REQUIRED)
include_directories(${FFTW_INCLUDES})

# TCLAP
find_package(TCLAP REQUIRED)
include_directories(${TCLAP_INCLUDES})

# CUDA
find_package(CUDA 7.5)
if (${CUDA_FOUND})
    find_package(GPUS)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Packages for the Python wrapper
option(HPP_BUILD_PYTHON "Build the Python interface to the library." TRUE)
if (${HPP_BUILD_PYTHON})
    set(Python_ADDITIONAL_VERSIONS "3.7" "3.6" "3.5") # in order of preference
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
    find_package(PythonExtras)
    find_package(Boost REQUIRED COMPONENTS python)
    include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
endif()

# FIND STANDARD LIBRARY FUNCTIONS #
###################################
# Some of these aren't available with certain combinations of compiler and
# standard library implementation, so we provide our own implementation
# in that case.

CHECK_SYMBOL_EXISTS(aligned_alloc stdlib.h HPP_HAVE_ALIGNED_ALLOC)

# OPTIONS #
###########

# CUDA
option(HPP_USE_CUDA "Use CUDA." ${CUDA_FOUND})
set(HPP_CUDA_ARCH_LIST ${CUDA_ARCHS_FOUND} CACHE string "List of CUDA architectures to use.")
option(HPP_SHOW_PTX "Show PTX information when compiling CUDA. (Developers)" OFF)

# CONFIGURATION HEADER #
########################
configure_file(${PROJECT_SOURCE_DIR}/include/hpp/config.h.in ${PROJECT_SOURCE_DIR}/include/hpp/config.h)

# INTERNALS #
#############

# Default output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/hpp)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/hpp)

# Pull the architectures-specific compiler flags into CPU_ARCHITECTURE_FLAGS
include(OptimizeForArchitecture)
OptimizeForArchitecture()
# Convert from ";" separated list to space separated string
string(REPLACE ";" " " CPU_ARCHITECTURE_FLAGS "${CPU_ARCHITECTURE_FLAGS}")

# Add the architectures-specific flags to all build configurations
set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} ${CPU_ARCHITECTURE_FLAGS}")

# Add OpenMP flags
set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Add general flags to both builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${COMMON_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COMMON_CXX_FLAGS}")

# Flags specific to debug and release builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -DDEBUG_BUILD -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")

# CUDA flags
if (${HPP_USE_CUDA})
    # NVCC common flags    
    set(COMMON_NVCC_FLAGS "${COMMON_NVCC_FLAGS} -std=c++11")
    if(${HPP_SHOW_PTX})
        set(COMMON_NVCC_FLAGS "${COMMON_NVCC_FLAGS} -Xptxas -v")
    endif(${HPP_SHOW_PTX})
    set(COMMON_NVCC_FLAGS "${COMMON_NVCC_FLAGS} -Xcompiler -fPIC -lineinfo")

    # Lowest possible virtual architecture for generating PTX
    set(COMMON_NVCC_FLAGS "${COMMON_NVCC_FLAGS} --gpu-architecture=compute_30")
    set(COMMON_NVCC_FLAGS "${COMMON_NVCC_FLAGS} --gpu-code=compute_30")
    
    # Architecture list to compile SASS for
    message(STATUS "Compiling for CUDA architectures: ${HPP_CUDA_ARCH_LIST}")
    set(COMMON_NVCC_FLAGS "${COMMON_NVCC_FLAGS} --gpu-code=${HPP_CUDA_ARCH_LIST}")

    # This is a workaround for an (as of 4 April 2017) unresolved GCC 5 + CUDA 7.5 interaction. 
    # See https://github.com/tensorflow/tensorflow/issues/1066.
    set(COMMON_NVCC_FLAGS "${COMMON_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")

    # This is a workaround for another GCC 5 + CUDA 7.5 issue.
    # See https://groups.google.com/forum/#!msg/caffe-users/Tm3OsZBwN9Q/XKGRKNdmBAAJ
    set(COMMON_NVCC_FLAGS "${COMMON_NVCC_FLAGS} -D_FORCE_INLINES")

    # Add the common flags
    set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} ${COMMON_NVCC_FLAGS}")
    set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} ${COMMON_NVCC_FLAGS}")

    # NVCC build-specific flags
    set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -g -G")
    # set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3 -use_fast_math")
    set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3")
endif(${HPP_USE_CUDA})

# Testing
enable_testing()

# Main subdirectories
# Things added after these will not be propagated down
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)

# Report
message(STATUS "Options:")
message(STATUS "")
message(STATUS "  CPU options:")
message(STATUS "    Architecture flags: ${CPU_ARCHITECTURE_FLAGS}")
message(STATUS "")
message(STATUS "  GPU options:")
if (${HPP_USE_CUDA})
    message(STATUS "    CUDA: ${HPP_USE_CUDA} (ver ${CUDA_VERSION_STRING})")
else()
    message(STATUS "    CUDA: ${HPP_USE_CUDA}")
endif()
# Installation of cmake configuration information
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/hpp/HPPConfigVersion.cmake"
  VERSION ${hpp_VERSION}
  COMPATIBILITY AnyNewerVersion
)
export(EXPORT hppTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/hpp/HPPTargets.cmake"
  NAMESPACE hpp::
)
configure_file("cmake/HPPConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hpp/HPPConfig.cmake"
  COPYONLY
)
set(ConfigPackageLocation "lib/cmake/hpp")
install(EXPORT hppTargets
        FILE HPPTargets.cmake
        NAMESPACE hpp::
        DESTINATION ${ConfigPackageLocation}
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/hpp/HPPConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/hpp/HPPConfigVersion.cmake"
  DESTINATION
    ${ConfigPackageLocation}
  COMPONENT
    Devel
)

