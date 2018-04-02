# Crystal tests
add_executable(testRotation testRotation.cpp)
target_link_libraries(testRotation hpp)
add_test(testRotation testRotation)