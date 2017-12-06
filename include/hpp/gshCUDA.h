/// @file gsh.h
/// @author Michael Malahe
/// @brief Header file for generalized spherical harmonic basis
#ifndef HPP_GSH_H
#define HPP_GSH_H

#include <hpp/config.h>

#ifdef HPP_USE_CUDA
    #include <cuComplex.h>
    #include <hpp/cudaUtils.h>
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
 * @tparam The real scalar type
 */
template <typename T>
class GSHCoeffsCUDA {
    public:
        __host__ __device__ GSHCoeffsCUDA(){
            for (unsigned int i=0; i<nl0; i++) {
                l0[i].x = 0.0;
                l0[i].y = 0.0;
            }
            for (unsigned int i=0; i<nl1; i++) {
                l1[i].x = 0.0;
                l1[i].y = 0.0;
            }
            for (unsigned int i=0; i<nl2; i++) {
                l2[i].x = 0.0;
                l2[i].y = 0.0;
            }
            for (unsigned int i=0; i<nl3; i++) {
                l3[i].x = 0.0;
                l3[i].y = 0.0;
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
        __host__ __device__ typename cuTypes<T>::complex get(int l, int m, int n) {
            unsigned int flatIdx = this->getFlatIdx(l,m,n);
            
            // Fetch the value
            typename cuTypes<T>::complex val;
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
                case 3:
                    val = l3[flatIdx];
                    break;
                default:
                    return;
            }
            
            // Modify for symmetry if in symmetrized section
            if (this->isInSymmetrizedSection(l,m,n)) {
                typename cuTypes<T>::complex mult;
                mult.x = pow(-1.0, m+n);
                mult.y = 0.0;
                val = mult*cuConj(val);
            }
            
            // Return
            return val;
            
        }
        
        ///@todo a way of indicating failure on l too large
        __host__ __device__ void set(int l, int m, int n, typename cuTypes<T>::complex val) {
            unsigned int flatIdx = this->getFlatIdx(l,m,n);
            
            // Modify for symmetry if in symmetrized section
            if (this->isInSymmetrizedSection(l,m,n)) {
                typename cuTypes<T>::complex mult;
                mult.x = pow(-1.0, m+n);
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
                case 3:
                    l3[flatIdx] = val;
                    break;
                default:
                    return;
            }
        }
        
        __host__ std::vector<T> getl0Reals() {
            std::vector<T> vals(2*nl0);
            for (unsigned int i=0; i<nl0; i++) {
                vals[2*i] = l0[i].x;
                vals[2*i+1] = l0[i].y;                
            }
            return vals;
        }
        
        __host__ std::vector<T> getl1Reals() {
            std::vector<T> vals(2*nl1);
            for (unsigned int i=0; i<nl1; i++) {
                vals[2*i] = l1[i].x;
                vals[2*i+1] = l1[i].y;                
            }
            return vals;
        }
        
        __host__ std::vector<T> getl2Reals() {
            std::vector<T> vals(2*nl2);
            for (unsigned int i=0; i<nl2; i++) {
                vals[2*i] = l2[i].x;
                vals[2*i+1] = l2[i].y;
            }
            return vals;
        }
        
        __host__ std::vector<T> getl3Reals() {
            std::vector<T> vals(2*nl3);
            for (unsigned int i=0; i<nl3; i++) {
                vals[2*i] = l3[i].x;
                vals[2*i+1] = l3[i].y;
            }
            return vals;
        }
        
        int nl0 = 2*0*(0+1)+1;
        int nl1 = 2*1*(1+1)+1;
        int nl2 = 2*2*(2+1)+1;
        int nl3 = 2*3*(3+1)+1;
        typename cuTypes<T>::complex l0[2*0*(0+1)+1];
        typename cuTypes<T>::complex l1[2*1*(1+1)+1];
        typename cuTypes<T>::complex l2[2*2*(2+1)+1];
        typename cuTypes<T>::complex l3[2*3*(3+1)+1];     
};

template <typename T>
__host__ __device__ GSHCoeffsCUDA<T> operator+(const GSHCoeffsCUDA<T>& coeffs1, const GSHCoeffsCUDA<T>& coeffs2) {
    GSHCoeffsCUDA<T> res;
    for (unsigned int i=0; i<res.nl0; i++) {
        res.l0[i] = coeffs1.l0[i]+coeffs2.l0[i];
    }
    for (unsigned int i=0; i<res.nl1; i++) {
        res.l1[i] = coeffs1.l1[i]+coeffs2.l1[i];
    }
    for (unsigned int i=0; i<res.nl2; i++) {
        res.l2[i] = coeffs1.l2[i]+coeffs2.l2[i];
    }
    for (unsigned int i=0; i<res.nl3; i++) {
        res.l3[i] = coeffs1.l3[i]+coeffs2.l3[i];
    }
    return res;
}

template <typename T>
__host__ __device__ void operator+=(GSHCoeffsCUDA<T>& A, const GSHCoeffsCUDA<T>& B) {
    A = A+B;
}

template <typename T>
__host__ __device__ GSHCoeffsCUDA<T> operator/(const GSHCoeffsCUDA<T>& coeffs, T val) {
    GSHCoeffsCUDA<T> res;
    for (unsigned int i=0; i<res.nl0; i++) {
        res.l0[i] = coeffs.l0[i]/val;
    }
    for (unsigned int i=0; i<res.nl1; i++) {
        res.l1[i] = coeffs.l1[i]/val;
    }
    for (unsigned int i=0; i<res.nl2; i++) {
        res.l2[i] = coeffs.l2[i]/val;
    }
    for (unsigned int i=0; i<res.nl3; i++) {
        res.l3[i] = coeffs.l3[i]/val;
    }
    return res;
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
    for (unsigned int i=0; i<coeffs.nl0; i++) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l0[i].x += __shfl_down(coeffs.l0[i].x, offset);
        }
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l0[i].y += __shfl_down(coeffs.l0[i].y, offset);
        }
    }
    for (unsigned int i=0; i<coeffs.nl1; i++) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l1[i].x += __shfl_down(coeffs.l1[i].x, offset);
        }
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l1[i].y += __shfl_down(coeffs.l1[i].y, offset);
        }
    }
    for (unsigned int i=0; i<coeffs.nl2; i++) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l2[i].x += __shfl_down(coeffs.l2[i].x, offset);
        }
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l2[i].y += __shfl_down(coeffs.l2[i].y, offset);
        }
    }
    for (unsigned int i=0; i<coeffs.nl3; i++) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l3[i].x += __shfl_down(coeffs.l3[i].x, offset);
        }
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l3[i].y += __shfl_down(coeffs.l3[i].y, offset);
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

template <typename T>
__global__ void BLOCK_REDUCE_KEPLER_GSH_COEFFS(GSHCoeffsCUDA<T> *in, GSHCoeffsCUDA<T> *out, int nTerms) {
    GSHCoeffsCUDA<T> sum;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<nTerms; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSumGSHCoeffs(sum);
    if (threadIdx.x==0) {
        out[blockIdx.x]=sum;
    }
}

#endif /* HPP_USE_CUDA*/
    
} //END NAMESPACE HPP
#endif /* HPP_GSH_H */