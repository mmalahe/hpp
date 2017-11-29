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
        __host__ __device__ GSHCoeffsCUDA(){;}
        
        __host__ __device__ unsigned int getFlatIdx(int l, int m, int n) {
            // If lower triangular, switch to upper triangular section
            if (n<m) {
                n = -n;
                m = -m;
            }            
            unsigned int mIdx = m-l;
            unsigned int nIdx = n-l;           
            
            // mIdx and nIdx are indices in an LxL matrix
            unsigned int L = 2*l+1;
            
            // Start with the index of the final element
            unsigned int nElements = (L*(L+1))/2;            
            unsigned int idx = nElements-1;
            
            // Subtract the triangular numbers below and including our row
            idx -= ((L-mIdx)*(L-mIdx+1))/2;
            
            // Add our column offset
            idx += (L-nIdx+1);
            
            // Return
            return idx;
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
                case 3:
                    val = l3[flatIdx];
                    break;
                default:
                    return 
            }
            
            // Modify for symmetry if lower triangular
            if (n<m) {
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
            
            // Modify for symmetry if lower triangular
            if (n<m) {
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
                case 3:
                    l3[flatIdx] = val;
                    break;
                default:
                    return 
            }
        }

    private:
        T l0[1];
        T l1[6];
        T l2[15];
        T l3[40];
}
#endif /* HPP_USE_CUDA*/
    
} //END NAMESPACE HPP
#endif /* HPP_GSH_H */