# Continuum tests
add_executable(testContinuum testContinuum.cpp)
target_link_libraries(testContinuum hpp)
add_test(testContinuum testContinuum)