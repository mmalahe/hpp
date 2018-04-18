add_executable(testCrystal testCrystal.cpp)
target_link_libraries(testCrystal hpp)
add_test(testCrystal testCrystal)

if (${HPP_USE_CUDA})
    CUDA_ADD_EXECUTABLE(testCrystalCUDA testCrystal.cu)
    target_link_libraries(testCrystalCUDA hpp)
endif()