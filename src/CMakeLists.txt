###########
# LIBRARY #
###########
set (HPP_CPP_SOURCES casesUtils.cpp tensor.cpp crystal.cpp hdfUtils.cpp spectralUtils.cpp mpiUtils.cpp profUtils.cpp)
if (${HPP_BUILD_PYTHON})
    set (HPP_CPP_SOURCES ${HPP_CPP_SOURCES} python.cpp)
endif()
set (HPP_CU_SOURCES cudaUtils.cu tensor.cu spectralUtils.cu crystal.cu)

if (${HPP_USE_CUDA})
    CUDA_ADD_LIBRARY(hpp ${HPP_CPP_SOURCES} ${HPP_CU_SOURCES})
else()
    add_library(hpp ${HPP_CPP_SOURCES})
endif()
    
target_link_libraries(hpp ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES})
target_link_libraries(hpp ${FFTW_LIBRARIES} ${FFTW_MPI_LIBRARIES})
target_link_libraries(hpp ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_link_libraries(hpp ${LAPACK_LIBRARIES})
target_link_libraries(hpp lapacke)
target_link_libraries(hpp ${BLAS_LIBRARIES})
target_link_libraries()

####################
# PYTHON INTERFACE #
####################
if (${HPP_BUILD_PYTHON})
    PYTHON_ADD_MODULE(hpppy python)
endif()

#########
# TOOLS #
#########
add_subdirectory(tools)
