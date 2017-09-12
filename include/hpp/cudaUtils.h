/** @file cudaUtils.h
* @author Michael Malahe
* @brief Header file CUDA utility functions
* @details
*/

#ifndef HPP_CUDA_UTILS_H
#define HPP_CUDA_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <memory>
#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <vector>
#include <hpp/config.h>

// Aligned memory
#ifdef __CUDACC__
    #define ALIGN(x)  __align__(x)
#else
    #if defined(__GNUC__)
        #define ALIGN(x)  __attribute__ ((aligned (x)))
    #else
        #define ALIGN(x)
    #endif
#endif

// Atomic add for doubles on CC < 6.0
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

namespace hpp
{
#ifdef HPP_USE_CUDA    

// Based on http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CUDA_CHK(ans) {cudaCheck((ans), __FILE__, __LINE__); }
inline void cudaCheck(cudaError_t code, const char *file, int line, bool abort=false){
    if (code != cudaSuccess){
        fprintf(stderr,"CUDA error at %s:%d -> %s\n", file, line, cudaGetErrorString(code));
        cudaGetLastError();
        if (abort) {
            throw std::runtime_error("CUDA error. See previous message.");
        }
    }
}

/**
 * @class CudaFreeDelete
 * @author Michael Malahe
 * @date 04/04/17
 * @file cudaUtils.h
 * @brief A simple deleter for cudaMalloc'd memory
 */
struct CudaFreeDelete {
    void operator()(void* x) {
        CUDA_CHK(cudaFree(x));
    }
};

template <typename T>
std::shared_ptr<T> cudaSharedPtr(T *devPtr) {
    return std::shared_ptr<T>(devPtr, CudaFreeDelete());
}

template <typename T>
T* allocDeviceMemory() {
    T *devPtr;
    CUDA_CHK(cudaMalloc((void**)&devPtr, sizeof(T)));
    return devPtr;
}

/**
 * @brief Allocate memory for this type
 * @return 
 */
template <typename T>
std::shared_ptr<T> allocDeviceMemorySharedPtr() {
    return cudaSharedPtr(allocDeviceMemory<T>());
}

template <typename T>
T* allocDeviceMemory(size_t n) {
    T *devPtr;
    CUDA_CHK(cudaMalloc((void**)&devPtr, n*sizeof(T)));
    return devPtr;
}

template <typename T>
std::shared_ptr<T> allocDeviceMemorySharedPtr(size_t n) {
    return cudaSharedPtr(allocDeviceMemory<T>(n));
}

template <typename T>
T *makeDeviceCopy(const T& hostVal) {
    T *devPtr = allocDeviceMemory<T>();
    CUDA_CHK(cudaMemcpy(devPtr, &hostVal, sizeof(T), cudaMemcpyHostToDevice));
    return devPtr;
}

template <typename T>
std::shared_ptr<T> makeDeviceCopySharedPtr(const T& hostVal) {
    std::shared_ptr<T> devPtr = allocDeviceMemorySharedPtr<T>();
    CUDA_CHK(cudaMemcpy(devPtr.get(), &hostVal, sizeof(T), cudaMemcpyHostToDevice));
    return devPtr;
}

template <typename T>
std::shared_ptr<T> makeDeviceCopySharedPtrFromPtr(const T* hostPtr) {
    std::shared_ptr<T> devPtr = allocDeviceMemorySharedPtr<T>();
    CUDA_CHK(cudaMemcpy(devPtr.get(), hostPtr, sizeof(T), cudaMemcpyHostToDevice));
    return devPtr;
}

/**
 * @brief Get the value of the host variable from a device pointer
 * @param devPtr
 * @return 
 */
template <typename T>
T getHostValue(const std::shared_ptr<T>& devPtr) {
    T hostVal;
    CUDA_CHK(cudaMemcpy((void*)&hostVal, devPtr.get(), sizeof(T), cudaMemcpyDeviceToHost));
    return hostVal;
}

template <typename T>
T getHostValue(T *devPtr) {
    T hostVal;
    CUDA_CHK(cudaMemcpy((void*)&hostVal, devPtr, sizeof(T), cudaMemcpyDeviceToHost));
    return hostVal;
}

template <typename T>
void copyToHost(const std::shared_ptr<T>& devPtr, T *hostPtr) {
    CUDA_CHK(cudaMemcpy((void*)hostPtr, devPtr.get(), sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T, typename A>
T *makeDeviceCopyVec(const std::vector<T,A>& vec) {
    size_t size = vec.size();
    size_t memSize = size*sizeof(T);
    T *devPtr;
    CUDA_CHK(cudaMalloc((void**)&devPtr, memSize));
    CUDA_CHK(cudaMemcpy(devPtr, vec.data(), memSize, cudaMemcpyHostToDevice));
    return devPtr;
}

template<typename T, typename A>
std::shared_ptr<T> makeDeviceCopyVecSharedPtr(const std::vector<T,A>& vec) {
    return cudaSharedPtr(makeDeviceCopyVec(vec));
}

template <typename T>
T *makeDevCopyOfDevArray(T *devPtrIn, size_t n) {
    size_t memSize = n*sizeof(T);
    T *devPtrCopy;
    CUDA_CHK(cudaMalloc((void**)&devPtrCopy, memSize));
    CUDA_CHK(cudaMemcpy(devPtrCopy, devPtrIn, memSize, cudaMemcpyDeviceToDevice));
    return devPtrCopy;
}

struct CudaKernelConfig {
    dim3 dB = {0,0,0};
    dim3 dG = {0,0,0};
    float occupancy;
};

unsigned int maxResidentWarps(const cudaDeviceProp& devProp);

CudaKernelConfig getKernelConfigMaxOccupancy(const cudaDeviceProp& devProp, const void *kernelPtr, unsigned int nThreads);

std::ostream& operator<<(std::ostream& out, const CudaKernelConfig& cfg);

// Mathematical function wrappers
// For functionIntrinsic, only calls it if it exists, otherwise defaults
// to non-intrinsic version

__inline__ __device__ float sinIntrinsic(float x) {
    return __sinf(x);
}
__inline__ __device__ double sinIntrinsic(double x) {
    return sin(x);
}

__inline__ __device__ void sincosIntrinsic(float a, float *b, float *c) {
    __sincosf(a,b,c);
}
__inline__ __device__ void sincosIntrinsic(double a, double *b, double *c) {
    sincos(a,b,c);
}

__inline__ __device__ void sincosFull(float a, float *b, float *c) {
    sincosf(a,b,c);
}
__inline__ __device__ void sincosFull(double a, double *b, double *c) {
    sincos(a,b,c);
}

__inline__ __device__ float powIntrinsic(float a, float b) {
    return __powf(a,b);
}
__inline__ __device__ double powIntrinsic(double a, double b) {
    return pow(a,b);
}

__inline__ __device__ float fmaIntrinsic(float x, float y, float z) {
    return __fmaf_rd(x,y,z);
}
__inline__ __device__ double fmaIntrinsic(double x, double y, double z) {
    return __fma_rd(x,y,z);
}

__inline__ __device__ float sqrtIntrinsic(float x) {
    return sqrtf(x);
}
__inline__ __device__ double sqrtIntrinsic(double x) {
    return sqrt(x);
}


/**
 * @brief Very rudimentary, but faster than intrinsic conversion
 * @param x
 * @return 
 */
__inline__ __device__ unsigned int cvtFloatToUint(float x) {
    // Cast memory
    unsigned int raw = *((int*)&x);
    // Get exponent
    unsigned int exponent = ((raw&2139095040)>>23)-127;
    // Mask the significand
    unsigned int significand = raw&8388607;
    // Prepend the 1
    significand |= 8388608;
    // Recompose
    unsigned int val = significand>>(23-exponent);
    // Return
    return val;
}
__inline__ __host__ __device__ unsigned int log2u(unsigned int val) {
    int output = 0;
    while (val >>= 1) ++output;
    return output;
}

/**
 * @brief 
 * @details Note that this is valid only on CC 3.0 and above. 
 * Taken from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 * @param val
 * @return 
 */
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    const int warpSize = 32;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

/**
 * @brief
 * @details Taken from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 * @param val
 */
template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
    const int warpSize = 32;
    static __shared__ T shared[warpSize]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

/**
 * @brief 
 * @details Taken from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 * @param in
 * @param out
 * @param N
 */
template <typename T>
__global__ void BLOCK_REDUCE_KEPLER(T *in, T* out, int N) {
    T sum = (T)0.0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<N; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x==0) {
        out[blockIdx.x]=sum;
    }
}

inline size_t getUsedMemoryBytes() {
    size_t freeBytes;
    size_t totalBytes;
    CUDA_CHK(cudaMemGetInfo(&freeBytes, &totalBytes));
    size_t usedBytes = totalBytes-freeBytes;
    return usedBytes;
}

inline double getUsedMemoryGB() {
    size_t usedBytes = getUsedMemoryBytes();
    double usedGB = ((double)usedBytes)/(1024*1024*1024);
    return usedGB;
}

inline double getUsedMemoryGiB() {
    size_t usedBytes = getUsedMemoryBytes();
    double usedGiB = ((double)usedBytes)/(1000*1000*1000);
    return usedGiB;
}

inline double getClockRateGHz(int deviceID) {
    cudaDeviceProp devProp;
    CUDA_CHK(cudaGetDeviceProperties(&devProp, deviceID));
    return ((double)devProp.clockRate)/1000000.0;
}

#endif /* HPP_USE_CUDA */
}//END NAMESPACE HPP

#endif /* HPP_CUDA_UTILS_H */