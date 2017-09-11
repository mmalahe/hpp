# Spectral utils tests
add_executable(testSpectralUtils testSpectralUtils.cpp)
target_link_libraries(testSpectralUtils hpp)
add_test(testSpectralUtils testSpectralUtils)
if (${HPP_USE_CUDA})
    CUDA_ADD_EXECUTABLE(testSpectralUtilsCUDA testSpectralUtilsCUDA.cu)
    target_link_libraries(testSpectralUtilsCUDA hpp)
    add_test(testSpectralUtilsCUDA testSpectralUtilsCUDA)
endif()