# HDFUtils parallel tests
add_executable(testHDFUtilsParallel testHDFUtilsParallel.cpp)
target_link_libraries(testHDFUtilsParallel hpp)
add_test(testHDFUtilsParallel mpirun -np 4 --allow-run-as-root testHDFUtilsParallel)
