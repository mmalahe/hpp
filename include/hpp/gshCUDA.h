/// @file gsh.h
/// @author Michael Malahe
/// @brief Header file for generalized spherical harmonic basis
#ifndef HPP_GSH_H
#define HPP_GSH_H

#include <hpp/config.h>

#ifdef HPP_USE_CUDA
    #include <cuComplex.h>
    #include <cudaUtils.h>
#endif

namespace hpp
{

#ifdef HPP_USE_CUDA
/**
 * @class GSHCoeffsCUDA
 * @author Michael Malahe
 * @date 29/11/17
 * @file gsh.h
 * @brief The complex generalized spherical harmonic coefficients for real data.
 * @detail For real data, there is the guarantee that 
 * C[l,-m,-n] = (-1)^{(m+n)} C[l,m,n]*. So, we only store the upper triangular
 * values for m and n, such that n>=m. These values are then flattened in
 * row major order.
 * @tparam The complex type. Either cuFloatComplex or cuDoubleComplex.
 */
template <typename T>
class GSHCoeffsCUDA {
    public:
        __host__ __device__ GSHCoeffsCUDA(){
            for (unsigned int i=0; i<1; i++) {
                l0[i].x = 0.0;
                l0[i].y = 0.0;
            }
            for (unsigned int i=0; i<5; i++) {
                l1[i].x = 0.0;
                l1[i].y = 0.0;
            }
            for (unsigned int i=0; i<13; i++) {
                l2[i].x = 0.0;
                l2[i].y = 0.0;
            }
        }
        
        __host__ __device__ bool isInSymmetrizedSection(int l, int m, int n) {
            int L = 2*l+1;
            int nElements = (L*L+1)/2;
            int mIdx = m+l;
            int nIdx = n+l;
            if (mIdx*L + nIdx > nElements-1) {
                return true;
            }
            else {
                return false;
            }
        }
        
        __host__ __device__ unsigned int getFlatIdx(int l, int m, int n) {
            // If lower triangular, switch to upper triangular section
            if (this->isInSymmetrizedSection(l,m,n)) {
                n = -n;
                m = -m;
            }            
            unsigned int mIdx = m+l;
            unsigned int nIdx = n+l;           
            
            // mIdx and nIdx are indices in an LxL matrix
            unsigned int L = 2*l+1;
            
            // Return
            return mIdx*L + nIdx;
        }
        
        ///@todo a way of indicating failure on l too large
        __host__ __device__ T get(int l, int m, int n) {
            unsigned int flatIdx = this->getFlatIdx(l,m,n);
            
            // Fetch the value
            T val;
            switch (l) {
                case 0:
                    val = l0[flatIdx];
                    break;
                case 1:
                    val = l1[flatIdx];
                    break;
                case 2:
                    val = l2[flatIdx];
                    break;
                default:
                    return 
            }
            
            // Modify for symmetry if in symmetrized section
            if (this->isInSymmetrizedSection(l,m,n)) {
                T mult;
                mult.x = powIntrinsic(-1.0, m+n);
                mult.y = 0.0;
                val = mult*cuConj(val);
            }
            
            // Return
            return val;
            
        }
        
        ///@todo a way of indicating failure on l too large
        __host__ __device__ void set(int l, int m, int n, T val) {
            unsigned int flatIdx = this->getFlatIdx(l,m,n);
            
            // Modify for symmetry if in symmetrized section
            if (this->isInSymmetrizedSection(l,m,n)) {
                T mult;
                mult.x = powIntrinsic(-1.0, m+n);
                mult.y = 0.0;
                val = mult*cuConj(val);
            }
            
            // Set value
            switch (l) {
                case 0:
                    l0[flatIdx] = val;
                    break;
                case 1:
                    l1[flatIdx] = val;
                    break;
                case 2:
                    l2[flatIdx] = val;
                    break;
                default:
                    return 
            }
        }
    
        T l0[1];
        T l1[5];
        T l2[13];        
}

// PARALLEL REDUCTION //
/**
 * @brief 
 * @details Note that this is valid only on CC 3.0 and above. 
 * Adapted from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 * @param val
 * @return 
 */
template <typename T>
inline __device__ GSHCoeffsCUDA<T> warpReduceSumGSHCoeffs(GSHCoeffsCUDA<T> coeffs) {
    const int warpSize = 32;
    for (unsigned int i=0; i<1; i++) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l0[i].x += __shfl_down(coeffs.l0[i].x, offset);
            coeffs.l0[i].y += __shfl_down(coeffs.l0[i].y, offset);
        }
    }
    for (unsigned int i=0; i<5; i++) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l1[i].x += __shfl_down(coeffs.l1[i].x, offset);
            coeffs.l1[i].y += __shfl_down(coeffs.l1[i].y, offset);
        }
    }
    for (unsigned int i=0; i<13; i++) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l2[i].x += __shfl_down(coeffs.l2[i].x, offset);
            coeffs.l2[i].y += __shfl_down(coeffs.l2[i].y, offset);
        }
    }
    return coeffs;
}

/**
 * @brief
 * @details Adapted from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 * @param val
 */
template <typename T>
inline __device__ GSHCoeffsCUDA<T> blockReduceSumGSHCoeffs(GSHCoeffsCUDA<T> val) {
    const int warpSize = 32;
    static __shared__ GSHCoeffsCUDA<T> shared[warpSize]; // Shared mem for 32 partial sums
    __syncthreads();
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSumGSHCoeffs(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    if (threadIdx.x < blockDim.x / warpSize) {
        val = shared[lane];
    }
    else {
        val = GSHCoeffsCUDA<T>();
    }

    if (wid==0) val = warpReduceSumGSHCoeffs(val); //Final reduce within first warp

    return val;
}

#endif /* HPP_USE_CUDA*/
    
} //END NAMESPACE HPP
#endif /* HPP_GSH_H */