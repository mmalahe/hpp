#include <hpp/cudaUtils.h>

namespace hpp
{
#ifdef HPP_USE_CUDA

/**
 * @brief Get the maximum number of resident warps per multiprocessor
 * @detail See http://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
 * @param devProp
 * @return 
 */
int getMaxResidentWarps(const cudaDeviceProp& devProp) {
    int cc = 10*devProp.major + devProp.minor;
    int maxWarps = 0;
    if (cc < 30) {
        maxWarps = 48;
    }
    else if (cc <= 62) {
        maxWarps = 64;
    }
    else {
        std::cerr << "WARNING: This function was written when 6.2 was the maximum compute cabability." << std::endl;
        std::cerr << "Assuming that the maximum number of warps for upcoming architectures is 64." << std::endl;
        maxWarps = 64;
    }
    return maxWarps;
}

CudaKernelConfig getKernelConfigMaxOccupancy(const cudaDeviceProp& devProp, const void *kernelPtr, unsigned int nThreads) {
    // Minimum grid size for full occupancy
    int minGridSize;
    
    // Recommended block size
    int blockSize;
    
    // Get parameters
    CUDA_CHK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelPtr));
    
    // Set up config
    CudaKernelConfig cfg;
    
    // Set the block dimensions
    cfg.dB.x = blockSize;
    cfg.dB.y = 1;
    cfg.dB.z = 1;    
    
    // Determine overall grid size
    int gridSize = nThreads/blockSize;
    if (nThreads % blockSize != 0) gridSize++;
    
    // See if we can fit the grid size into the first dimension
    // This should only be a concern with compute capability
    // 2.x and lower and very large grids, as the maximum x grid dimension there
    // is 65535, while it is 2^31-1 for subsequent architectures.
    if (gridSize <= devProp.maxGridSize[0]) {
        cfg.dG.x = gridSize;
        cfg.dG.y = 1;
        cfg.dG.z = 1;
    }
    else {
        ///@todo Either remove 2.x support, or implement distributing the grid size over multiple dimensions
        throw std::runtime_error("No implementation for ncrystal greater than about 50m for compute capability 2.x and lower.");
    }
    
    // Determine the occupancy
    int maxBlocksPerMP;
    size_t dynamicSMemSize = 0;
    CUDA_CHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMP, kernelPtr, blockSize, dynamicSMemSize));
    int maxWarpsPerMP = maxBlocksPerMP*blockSize/devProp.warpSize;
    int maxResidentWarps = getMaxResidentWarps(devProp);
    cfg.occupancy = ((float)maxWarpsPerMP)/maxResidentWarps;
    
    // Return
    return cfg;
}

std::ostream& operator<<(std::ostream& out, const CudaKernelConfig& cfg)
{
    out << "dB: " << cfg.dB.x << " " << cfg.dB.y << " " << cfg.dB.z << std::endl;
    out << "dG: " << cfg.dG.x << " " << cfg.dG.y << " " << cfg.dG.z << std::endl;
    out << "Theoretical occupancy: " << 100*cfg.occupancy << "%" << std::endl;
    return out;
}

// Functions for testing or inspecting compiled products
__global__ void TEST_IFMA(int *a, int *b, int *c) 
{
    c[threadIdx.x] += a[threadIdx.x]*b[threadIdx.x];
}

__global__ void TEST_IMUL(int *a, int *b, int *c) 
{
    c[threadIdx.x] = a[threadIdx.x]*b[threadIdx.x];
}

__global__ void TEST_I_TIMES_F(int *a, float *b, float *c) 
{
    c[threadIdx.x] = a[threadIdx.x]*b[threadIdx.x];
}

__global__ void TEST_DSIN(double *x, double *s) {
    s[threadIdx.x] = sinIntr(x[threadIdx.x]);
}

__global__ void TEST_I2F(int *a, float *b) {
    b[threadIdx.x] = (float)a[threadIdx.x];
}

#endif /* HPP_USE_CUDA */
}//END NAMESPACE HPP