# Crystal tests
add_executable(testCrystal testCrystal.cpp)
target_link_libraries(testCrystal hpp)
add_test(testCrystal testCrystal)