# FFTW serial tests
add_executable(testFFTWSerial testFFTWSerial.cpp)
target_link_libraries(testFFTWSerial hpp)
add_test(testFFTWSerial testFFTWSerial)

# FFTW parallel tests
add_executable(testFFTWParallel testFFTWParallel.cpp)
target_link_libraries(testFFTWParallel hpp)
add_test(testFFTWParallel mpirun -np 4  --allow-run-as-root testFFTWParallel)
