# Tensor2 tests
add_executable(testTensor2 testTensor2.cpp)
target_link_libraries(testTensor2 hpp)
add_test(testTensor2 testTensor2)

# Tensor2 parallel tests
add_executable(testTensor2Parallel testTensor2Parallel.cpp)
target_link_libraries(testTensor2Parallel hpp)
add_test(testTensor2Parallel mpirun -np 4 --allow-run-as-root testTensor2Parallel)

# Tensor4 tests
add_executable(testTensor4 testTensor4.cpp)
target_link_libraries(testTensor4 hpp)
add_test(testTensor4 testTensor4)

# Remaining tensor tests
add_executable(testTensor testTensor.cpp)
target_link_libraries(testTensor hpp)
add_test(testTensor testTensor)