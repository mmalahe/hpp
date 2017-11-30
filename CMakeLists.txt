cmake_minimum_required (VERSION 3.4)

# Project name and version.
project (hpp)
set (hpp_VERSION_MAJOR 0)
set (hpp_VERSION_MINOR 5)

# COMPILERS #
#############
set(CMAKE_CXX_STANDARD 11)

# MODULES #
###########

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

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

# SSE instructions
find_package(SSE)

# CUDA
find_package(CUDA 7.5)
if (${CUDA_FOUND})
    find_package(GPUS)
    include_directories(${CUDA_INCLUDE_DIRS})
    message(${CUDA_INCLUDE_DIRS})
endif()

# Packages for the Python wrapper
option(HPP_BUILD_PYTHON "Build the Python interface to the library." TRUE)
if (${HPP_BUILD_PYTHON})
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

# SSE instructions
option(HPP_USE_MMX "Use MMX instructions." ${MMX_FOUND})
option(HPP_USE_SSE "Use SSE instructions." ${SSE_FOUND})
option(HPP_USE_SSE2 "Use SSE2 instructions." ${SSE2_FOUND})
option(HPP_USE_SSE3 "Use SSE3 instructions." ${SSE3_FOUND})
option(HPP_USE_SSE4_1 "Use SSE4.1 instructions." ${SSE4_1_FOUND})
option(HPP_USE_SSE4_2 "Use SSE4.2 instructions." ${SSE4_2_FOUND})
option(HPP_USE_AVX "Use AVX instructions." ${AVX_FOUND})
option(HPP_USE_AVX2 "Use AVX2 instructions." ${AVX2_FOUND})
option(HPP_USE_FMA "Use FMA instructions." ${FMA_FOUND})
option(HPP_USE_FMA4 "Use FMA4 instructions." ${FMA4_FOUND})

# CUDA
option(HPP_USE_CUDA "Use CUDA." ${CUDA_FOUND})
set(HPP_CUDA_ARCH_LIST ${CUDA_ARCHS_FOUND} CACHE string "List of CUDA architectures to use.")

# CONFIGURATION HEADER #
########################
configure_file(${PROJECT_SOURCE_DIR}/include/hpp/config.h.in ${PROJECT_SOURCE_DIR}/include/hpp/config.h)

# INTERNALS #
#############

# Intra-project directories
include_directories(include)

# Default output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/hpp)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/hpp)

# Generate additional flags for both debug and release builds
set(COMMON_CXX_FLAGS "${HPP_GENERAL_CXX_FLAGS} -ggdb ${OpenMP_CXX_FLAGS}")

# Compiler-specific flags
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(STATUS "Adding GNU/Clang compiler flags.")
    # SSE instructions
    if (${HPP_USE_MMX})
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -mmmx")
    endif()
    if (${HPP_USE_SSE})
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -msse")
    endif()
    if (${HPP_USE_SSE2})
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -msse2")
    endif()
    if (${HPP_USE_SSE3})
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -msse3")
    endif()
    if (${HPP_USE_SSE4_1})
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -msse4.1")
    endif()
    if (${HPP_USE_SSE4_2})
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -msse4.2")
    endif()
    if (${HPP_USE_AVX})
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -mavx")
    endif()
    if (${HPP_USE_AVX2})
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -mavx2")
    endif()
    if (${HPP_USE_FMA})
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -mfma")
    endif()
    if (${HPP_USE_FMA4})
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -mfma4")
    endif()
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message(STATUS "Using Clang-specific compiler flags.")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(STATUS "Using GNU-specific compiler flags.")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    message(STATUS "Using PGI-specific compiler flags.")
endif()

# Add general flags to both builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${COMMON_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COMMON_CXX_FLAGS}")

# Flags specific to debug and release builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG_BUILD -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# CUDA flags
if (${HPP_USE_CUDA})
    # NVCC common flags
    set(COMMON_NVCC_FLAGS "${COMMON_NVCC_FLAGS} -std=c++11 -Xptxas -v -Xcompiler -fPIC -lineinfo")

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
message(STATUS "    MMX: ${HPP_USE_MMX}")
message(STATUS "    SSE: ${HPP_USE_SSE}")
message(STATUS "    SSE2: ${HPP_USE_SSE2}")
message(STATUS "    SSE3: ${HPP_USE_SSE3}")
message(STATUS "    SSE4.1: ${HPP_USE_SSE4_1}")
message(STATUS "    SSE4.2: ${HPP_USE_SSE4_2}")
message(STATUS "    AVX: ${HPP_USE_AVX}")
message(STATUS "    AVX2: ${HPP_USE_AVX2}")
message(STATUS "    FMA: ${HPP_USE_FMA}")
message(STATUS "    FMA4: ${HPP_USE_FMA4}")
message(STATUS "")
message(STATUS "  GPU options:")
if (${HPP_USE_CUDA})
    message(STATUS "    CUDA: ${HPP_USE_CUDA} (ver ${CUDA_VERSION_STRING})")
else()
    message(STATUS "    CUDA: ${HPP_USE_CUDA}")
endif()