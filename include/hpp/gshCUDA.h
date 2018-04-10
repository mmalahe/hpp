/// @file gshCUDA.h
/// @author Michael Malahe
/// @brief Header file for generalized spherical harmonic basis
#ifndef HPP_GSHCUDA_H
#define HPP_GSHCUDA_H

#include <hpp/config.h>
HPP_CHECK_CUDA_ENABLED_BUILD
#include <cuComplex.h>
#include <hpp/cudaUtils.h>
#include <hpp/gsh.h>

namespace hpp
{

/**
 * @class GSHCoeffsCUDA
 * @author Michael Malahe
 * @date 29/11/17
 * @file gshCUDA.h
 * @brief The complex generalized spherical harmonic coefficients for real data.
 * @details For real data, there is the guarantee that 
 * C[l,-m,-n] = (-1)^{(m+n)} C[l,m,n]*. So, we only store the upper triangular
 * values for m and n, such that n>=m. These values are then flattened in
 * row major order.
 * @tparam The real scalar type
 */
template <typename T>
class GSHCoeffsCUDA {
    public:
        __host__ __device__ GSHCoeffsCUDA(){
            for (int i=0; i<nl0; i++) {
                l0[i].x = 0.0;
                l0[i].y = 0.0;
            }
            for (int i=0; i<nl1; i++) {
                l1[i].x = 0.0;
                l1[i].y = 0.0;
            }
            for (int i=0; i<nl2; i++) {
                l2[i].x = 0.0;
                l2[i].y = 0.0;
            }
            for (int i=0; i<nl3; i++) {
                l3[i].x = 0.0;
                l3[i].y = 0.0;
            }
            for (int i=0; i<nl4; i++) {
                l4[i].x = 0.0;
                l4[i].y = 0.0;
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
                case 4:
                    val = l4[flatIdx];
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
                case 4:
                    l4[flatIdx] = val;
                    break;
                default:
                    return;
            }
        }
        
        __host__ std::vector<T> getl0Reals() {
            std::vector<T> vals(2*nl0);
            for (int i=0; i<nl0; i++) {
                vals[2*i] = l0[i].x;
                vals[2*i+1] = l0[i].y;                
            }
            return vals;
        }
        
        __host__ std::vector<T> getl1Reals() {
            std::vector<T> vals(2*nl1);
            for (int i=0; i<nl1; i++) {
                vals[2*i] = l1[i].x;
                vals[2*i+1] = l1[i].y;                
            }
            return vals;
        }
        
        __host__ std::vector<T> getl2Reals() {
            std::vector<T> vals(2*nl2);
            for (int i=0; i<nl2; i++) {
                vals[2*i] = l2[i].x;
                vals[2*i+1] = l2[i].y;
            }
            return vals;
        }
        
        __host__ std::vector<T> getl3Reals() {
            std::vector<T> vals(2*nl3);
            for (int i=0; i<nl3; i++) {
                vals[2*i] = l3[i].x;
                vals[2*i+1] = l3[i].y;
            }
            return vals;
        }
        
        __host__ std::vector<T> getl4Reals() {
            std::vector<T> vals(2*nl4);
            for (int i=0; i<nl4; i++) {
                vals[2*i] = l4[i].x;
                vals[2*i+1] = l4[i].y;
            }
            return vals;
        }
        
        __host__ GSHCoeffsCUDA(const std::vector<T>& inputReals, int nLevels) {
            // Checks
            if (nLevels < 1 || nLevels > 5) {
                std::cerr << "nLevels = " << nLevels << std::endl;
                throw std::runtime_error("Number of levels should be betweeen 1 and 5.");
            }
            int nReals = inputReals.size();
            if (nReals != nGSHReals(nLevels)) {
                std::cerr << "inputReals.size() = " << nReals;
                std::cerr << "nGSHReals(nLevels) = " << nGSHReals(nLevels);
                throw std::runtime_error("Mismatch in number of real values for the number of GSH levels.");
            }            
            
            // Populate
            int offset = 0;
            if (nLevels >= 1) {
                for (int i=0; i<nl0; i++) {
                    int idx = offset+2*i;
                    l0[i].x = inputReals[idx];
                    l0[i].y = inputReals[idx+1];
                }
            }
            if (nLevels >= 2) {
                offset += nl0*2;
                for (int i=0; i<nl1; i++) {
                    int idx = offset+2*i;
                    l1[i].x = inputReals[idx];
                    l1[i].y = inputReals[idx+1];
                }
            }
            if (nLevels >= 3) {
                offset += nl1*2;
                for (int i=0; i<nl2; i++) {
                    int idx = offset+2*i;
                    l2[i].x = inputReals[idx];
                    l2[i].y = inputReals[idx+1];
                }
            }
            if (nLevels >= 4) {
                offset += nl2*2;
                for (int i=0; i<nl3; i++) {
                    int idx = offset+2*i;
                    l3[i].x = inputReals[idx];
                    l3[i].y = inputReals[idx+1];
                }
            }
            if (nLevels >= 5) {
                offset += nl3*2;
                for (int i=0; i<nl4; i++) {
                    int idx = offset+2*i;
                    l4[i].x = inputReals[idx];
                    l4[i].y = inputReals[idx+1];
                }
            }
        } 
        
        __host__ std::vector<T> getReals(unsigned int nLevels) {
            if (nLevels < 1 || nLevels > 5) {
                std::cerr << "nLevels = " << nLevels << std::endl;
                throw std::runtime_error("Number of levels should be betweeen 1 and 5.");
            }
            std::vector<T> reals = this->getl0Reals();
            if (nLevels >= 2) {
                auto level = this->getl1Reals();
                reals.insert(reals.end(), level.begin(), level.end());
            }
            if (nLevels >= 3) {
                auto level = this->getl2Reals();
                reals.insert(reals.end(), level.begin(), level.end());
            }
            if (nLevels >= 4) {
                auto level = this->getl3Reals();
                reals.insert(reals.end(), level.begin(), level.end());
            }
            if (nLevels >= 5) {
                auto level = this->getl4Reals();
                reals.insert(reals.end(), level.begin(), level.end());
            }
            return reals;
        }
        
        static const int nl0 = nGSHCoeffsInLevel(0);
        static const int nl1 = nGSHCoeffsInLevel(1);
        static const int nl2 = nGSHCoeffsInLevel(2);
        static const int nl3 = nGSHCoeffsInLevel(3);
        static const int nl4 = nGSHCoeffsInLevel(4);
        typename cuTypes<T>::complex l0[nl0];
        typename cuTypes<T>::complex l1[nl1];
        typename cuTypes<T>::complex l2[nl2];
        typename cuTypes<T>::complex l3[nl3];
        typename cuTypes<T>::complex l4[nl4];
};

template <typename T>
__host__ __device__ GSHCoeffsCUDA<T> operator+(const GSHCoeffsCUDA<T>& coeffs1, const GSHCoeffsCUDA<T>& coeffs2) {
    GSHCoeffsCUDA<T> res;
    for (int i=0; i<res.nl0; i++) {
        res.l0[i] = coeffs1.l0[i]+coeffs2.l0[i];
    }
    for (int i=0; i<res.nl1; i++) {
        res.l1[i] = coeffs1.l1[i]+coeffs2.l1[i];
    }
    for (int i=0; i<res.nl2; i++) {
        res.l2[i] = coeffs1.l2[i]+coeffs2.l2[i];
    }
    for (int i=0; i<res.nl3; i++) {
        res.l3[i] = coeffs1.l3[i]+coeffs2.l3[i];
    }
    for (int i=0; i<res.nl4; i++) {
        res.l4[i] = coeffs1.l4[i]+coeffs2.l4[i];
    }
    return res;
}

template <typename T>
__host__ __device__ void operator+=(GSHCoeffsCUDA<T>& A, const GSHCoeffsCUDA<T>& B) {
    A = A+B;
}

template <typename T>
__host__ __device__ GSHCoeffsCUDA<T> operator*(T val, const GSHCoeffsCUDA<T>& coeffs) {
    GSHCoeffsCUDA<T> res;
    for (int i=0; i<res.nl0; i++) {
        res.l0[i] = val*coeffs.l0[i];
    }
    for (int i=0; i<res.nl1; i++) {
        res.l1[i] = val*coeffs.l1[i];
    }
    for (int i=0; i<res.nl2; i++) {
        res.l2[i] = val*coeffs.l2[i];
    }
    for (int i=0; i<res.nl3; i++) {
        res.l3[i] = val*coeffs.l3[i];
    }
    for (int i=0; i<res.nl4; i++) {
        res.l4[i] = val*coeffs.l4[i];
    }
    return res;
}

template <typename T>
__host__ __device__ GSHCoeffsCUDA<T> operator/(const GSHCoeffsCUDA<T>& coeffs, T val) {
    GSHCoeffsCUDA<T> res;
    for (int i=0; i<res.nl0; i++) {
        res.l0[i] = coeffs.l0[i]/val;
    }
    for (int i=0; i<res.nl1; i++) {
        res.l1[i] = coeffs.l1[i]/val;
    }
    for (int i=0; i<res.nl2; i++) {
        res.l2[i] = coeffs.l2[i]/val;
    }
    for (int i=0; i<res.nl3; i++) {
        res.l3[i] = coeffs.l3[i]/val;
    }
    for (int i=0; i<res.nl4; i++) {
        res.l4[i] = coeffs.l4[i]/val;
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
    for (unsigned int i=0; i<coeffs.nl4; i++) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l4[i].x += __shfl_down(coeffs.l4[i].x, offset);
        }
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            coeffs.l4[i].y += __shfl_down(coeffs.l4[i].y, offset);
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
    
} //END NAMESPACE HPP
#endif /* HPP_GSHCUDA_H */