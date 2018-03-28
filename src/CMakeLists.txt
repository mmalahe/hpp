# CMake modules
include(HPPCommon)

#######################################################
# SOURCE FILES FROM INDEPENDENTLY DEVELOPED LIBRARIES #
#######################################################
add_subdirectory(external)

###############
# C++ LIBRARY #
###############
set (HPP_CPP_SOURCES casesUtils.cpp tensor.cpp crystal.cpp hdfUtils.cpp spectralUtils.cpp mpiUtils.cpp profUtils.cpp)
set (HPP_CPP_SOURCES ${HPP_CPP_SOURCES} ${EXTERNAL_CPP_SOURCES})
message(${EXTERNAL_CPP_SOURCES})
set (HPP_CU_SOURCES cudaUtils.cu tensor.cu spectralUtils.cu crystal.cu)

if (${HPP_USE_CUDA})
    CUDA_ADD_LIBRARY(hpp SHARED ${HPP_CPP_SOURCES} ${HPP_CU_SOURCES})
else()
    add_library(hpp SHARED ${HPP_CPP_SOURCES})
endif()

target_include_directories(hpp 
    PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>  
    $<INSTALL_INTERFACE:include>
)

# Internal libraries
target_link_libraries(hpp)

# External libraries
target_link_libraries(hpp ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES})
target_link_libraries(hpp ${FFTW_LIBRARIES} ${FFTW_MPI_LIBRARIES})
target_link_libraries(hpp ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_link_libraries(hpp ${LAPACK_LIBRARIES})
target_link_libraries(hpp lapacke)
target_link_libraries(hpp ${BLAS_LIBRARIES})

####################
# PYTHON INTERFACE #
####################
if (${HPP_BUILD_PYTHON})    
    if (${HPP_USE_CUDA})
        PYTHON_ADD_CUDA_MODULE(hpppy python.cu)
    else()
        PYTHON_ADD_MODULE(hpppy python)
    endif()        
    target_link_libraries(hpppy ${Boost_LIBRARIES})
    target_link_libraries(hpppy hpp)
endif()

#########
# TOOLS #
#########
add_subdirectory(tools)

# INSTALLATION #
################

# Installation of headers
install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/hpp"
    DESTINATION "include"
    FILES_MATCHING PATTERN "*.h"
)

# Installation of main library
install(TARGETS hpp
        EXPORT hppTargets
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib"
        INCLUDES DESTINATION "include"
)

# Installation of Python interface
if (${HPP_BUILD_PYTHON})
    install(
        TARGETS hpppy
        LIBRARY
        DESTINATION "${PYTHON_SITE_PACKAGES}"
    )
endif()