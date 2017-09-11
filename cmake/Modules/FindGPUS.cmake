# File to check architectures
set(__gpu_test_file "${PROJECT_BINARY_DIR}/gputest.cu")
file(WRITE ${__gpu_test_file} ""
  "#include <cstdio>\n"
  "int main()\n"
  "{\n"
  "  int count = 0;\n"
  "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
  "  if (count == 0) return -1;\n"
  "  for (int device = 0; device < count; ++device)\n"
  "  {\n"
  "    cudaDeviceProp prop;\n"
  "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device)){\n"
  "      std::printf(\"sm_%d%d\", prop.major, prop.minor);\n"
  "      if (device != count-1) std::printf(\",\");\n"
  "    }"
  "  }\n"
  "  return 0;\n"
  "}\n")

# Run file and get output
execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${__gpu_test_file}"
                WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                RESULT_VARIABLE __gpus_found OUTPUT_VARIABLE __gpu_cuda_archs
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Set variables
set(GPUS_FOUND ${__gpus_found} CACHE BOOL "GPUs were found")
set(CUDA_ARCHS_FOUND ${__gpu_cuda_archs} CACHE STRING "List of CUDA architectures found")
mark_as_advanced(GPUS_FOUND CUDA_ARCHS_FOUND)