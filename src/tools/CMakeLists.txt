# Put these tools in bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Iterative solver
add_executable(iterativeSolve iterativeSolve.cpp)
target_link_libraries(iterativeSolve hpp)

# Spectral solver
add_executable(generateRawDatabase generateRawDatabase.cpp)
target_link_libraries(generateRawDatabase hpp)

add_executable(generateSpectralDatabase generateSpectralDatabase.cpp)
target_link_libraries(generateSpectralDatabase hpp)

add_executable(evaluateCompressionError evaluateCompressionError.cpp)
target_link_libraries(evaluateCompressionError hpp)

add_executable(spectralSolve spectralSolve.cpp)
target_link_libraries(spectralSolve hpp)

if (${HPP_USE_CUDA})
    CUDA_ADD_EXECUTABLE(spectralSolveCUDA spectralSolveCUDA.cu)
    target_link_libraries(spectralSolveCUDA hpp)
    CUDA_ADD_EXECUTABLE(spectralSolveGSHCUDA spectralSolveGSHCUDA.cu)
    target_link_libraries(spectralSolveGSHCUDA hpp)
endif()

# Build directories for external tools to read
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    configure_file(${PROJECT_SOURCE_DIR}/config/debug.in ${PROJECT_SOURCE_DIR}/config/debug.txt)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    configure_file(${PROJECT_SOURCE_DIR}/config/release.in ${PROJECT_SOURCE_DIR}/config/release.txt)
endif()

