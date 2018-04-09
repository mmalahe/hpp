# mpiUtils tests
add_executable(testMPIUtils testMPIUtils.cpp)
target_link_libraries(testMPIUtils hpp)
add_test(testMPIUtils mpirun -np 3 --allow-run-as-root testMPIUtils)
