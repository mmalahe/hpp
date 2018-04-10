/// @file gshCUDA.h
/// @author Michael Malahe
/// @brief Header file for generalized spherical harmonic basis
#ifndef HPP_GSH_H
#define HPP_GSH_H

#include <hpp/config.h>
#include <cmath>
#include <complex>

namespace hpp
{

/**
 * @class GSHCoeffs
 * @author Michael Malahe
 * @date 29/11/17
 * @file gsh.h
 * @brief The complex generalized spherical harmonic coefficients for real data.
 * @details For real data, there is the guarantee that 
 * C[l,-m,-n] = (-1)^{(m+n)} C[l,m,n]*. So, we only store the upper triangular
 * values for m and n, such that n>=m. These values are then flattened in
 * row major order.
 * @tparam The real scalar type
 */
template <typename T>
class GSHCoeffs {
    public:
        GSHCoeffs(){
            for (int i=0; i<nl0; i++) {
                l0[i].real(0.0);
                l0[i].imag(0.0);
            }
            for (int i=0; i<nl1; i++) {
                l1[i].real(0.0);
                l1[i].imag(0.0);
            }
            for (int i=0; i<nl2; i++) {
                l2[i].real(0.0);
                l2[i].imag(0.0);
            }
            for (int i=0; i<nl3; i++) {
                l3[i].real(0.0);
                l3[i].imag(0.0);
            }
            for (int i=0; i<nl4; i++) {
                l4[i].real(0.0);
                l4[i].imag(0.0);
            }
        }
        
       bool isInSymmetrizedSection(int l, int m, int n) {
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
        
        unsigned int getFlatIdx(int l, int m, int n) {
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
        
        std::complex<T> get(int l, int m, int n) {
            unsigned int flatIdx = this->getFlatIdx(l,m,n);
            
            // Fetch the value
            std::complex<T> val;
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
                    throw std::runtime_error("No implementation for l>4");
            }
            
            // Modify for symmetry if in symmetrized section
            if (this->isInSymmetrizedSection(l,m,n)) {
                std::complex<T> mult;
                mult.real(std::pow(-1.0, m+n));
                mult.imag(0.0);
                val = mult*std::conj(val);
            }
            
            // Return
            return val;
            
        }
        
        void set(int l, int m, int n, std::complex<T> val) {
            unsigned int flatIdx = this->getFlatIdx(l,m,n);
            
            // Modify for symmetry if in symmetrized section
            if (this->isInSymmetrizedSection(l,m,n)) {
                std::complex<T> mult;
                mult.real(std::pow(-1.0, m+n));
                mult.imag(0.0);
                val = mult*std::conj(val);
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
                    throw std::runtime_error("No implementation for l>4");
            }
        }
        
        std::vector<T> getl0Reals() {
            std::vector<T> vals(2*nl0);
            for (int i=0; i<nl0; i++) {
                vals[2*i] = l0[i].real();
                vals[2*i+1] = l0[i].imag();                
            }
            return vals;
        }
        
        std::vector<T> getl1Reals() {
            std::vector<T> vals(2*nl1);
            for (int i=0; i<nl1; i++) {
                vals[2*i] = l1[i].real();
                vals[2*i+1] = l1[i].imag();                
            }
            return vals;
        }
        
        std::vector<T> getl2Reals() {
            std::vector<T> vals(2*nl2);
            for (int i=0; i<nl2; i++) {
                vals[2*i] = l2[i].real();
                vals[2*i+1] = l2[i].imag();
            }
            return vals;
        }
        
        std::vector<T> getl3Reals() {
            std::vector<T> vals(2*nl3);
            for (int i=0; i<nl3; i++) {
                vals[2*i] = l3[i].real();
                vals[2*i+1] = l3[i].imag();
            }
            return vals;
        }
        
        std::vector<T> getl4Reals() {
            std::vector<T> vals(2*nl4);
            for (int i=0; i<nl4; i++) {
                vals[2*i] = l4[i].real();
                vals[2*i+1] = l4[i].imag();
            }
            return vals;
        }
        
        int nl0 = 2*0*(0+1)+1;//1
        int nl1 = 2*1*(1+1)+1;//5
        int nl2 = 2*2*(2+1)+1;//13
        int nl3 = 2*3*(3+1)+1;//25
        int nl4 = 2*4*(4+1)+1;//41
        std::complex<T> l0[2*0*(0+1)+1];
        std::complex<T> l1[2*1*(1+1)+1];
        std::complex<T> l2[2*2*(2+1)+1];
        std::complex<T> l3[2*3*(3+1)+1];
        std::complex<T> l4[2*4*(4+1)+1];
};

constexpr int nGSHCoeffsInLevel(unsigned int iLevel) {
    return 2*iLevel*(iLevel+1)+1;
}

constexpr int nGSHReals(unsigned int nLevels) {
    return nLevels == 1 ? 2 : 
    (nLevels == 2 ? 12  : 
    (nLevels == 3 ? 38 : 
    (nLevels == 4 ? 88 : 
    (nLevels == 5 ? 129 : -1 
    ))));
}

template <typename T>
GSHCoeffs<T> operator+(const GSHCoeffs<T>& coeffs1, const GSHCoeffs<T>& coeffs2) {
    GSHCoeffs<T> res;
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
void operator+=(GSHCoeffs<T>& A, const GSHCoeffs<T>& B) {
    A = A+B;
}

template <typename T>
GSHCoeffs<T> operator/(const GSHCoeffs<T>& coeffs, const T val) {
    GSHCoeffs<T> res;
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

template <typename T>
void operator/=(GSHCoeffs<T>& A, const T B) {
    A = A/B;
}

} //END NAMESPACE HPP
#endif /* HPP_GSH_H */