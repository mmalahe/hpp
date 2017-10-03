/// @file continuum.h
/// @author Michael Malahe
/// @brief Header file for continuum mechanics classes and functions.
#ifndef HPP_CONTINUUM_H
#define HPP_CONTINUUM_H

#include <hpp/config.h>
#include <hpp/tensor.h>

// Units
// Quantities are in GPA
#define GPA (1.0)
#define MPA (1e-3)
#define KPA (1e-6)
#define PA (1e-9)

namespace hpp
{

template <typename U>
hpp::Tensor2<U> stretchingVelocityGradient(U theta, U eDot) {
    // D components
    std::vector<U> D_comps(3);
    D_comps[0] = std::sqrt(2.0/3.0)*std::cos(theta-M_PI/3.0);
    D_comps[1] = std::sqrt(2.0/3.0)*std::cos(theta+M_PI/3.0);
    D_comps[2] = -std::sqrt(2.0/3.0)*std::cos(theta);
    
    // Standard basis
    std::vector<std::vector<U>> basis(3);
    for (int i=0; i<3; i++) {
        basis[i] = std::vector<U>(3);
        basis[i][i] = 1.0;
    }
    
    // Construct D_0
    hpp::Tensor2<U> D_0(3,3);
    for (int i=0; i<3; i++) {
        std::vector<U> e_i = basis[i];
        D_0 += D_comps[i]*hpp::outer(e_i, e_i);
    }
    hpp::Tensor2<U> D = eDot*D_0;
    
    // Return
    return D;
}
    
template <typename T>
struct StretchingTensorDecomposition {
    T theta;
    T DNorm;
    Tensor2<T> evecs;
};
 
/**
* @brief 
* @param L the velocity gradient tensor
* @return 
*/
template <typename T>
StretchingTensorDecomposition<T> getStretchingTensorDecomposition(const hpp::Tensor2<T>& L) {
    // The decomposition
    StretchingTensorDecomposition<T> decomp;

    // Stretching tensor from velocity gradient
    hpp::Tensor2<T> D = ((T)0.5)*(L + L.trans());

    // Evec decomposition of D
    std::valarray<T> Devals;
    D.evecDecomposition(Devals, decomp.evecs);
    
    // Ensure that the determinant of the eigenvector matrix is positive
    // for the purposes of treating it like a rotation
    hpp::Tensor2<T> RTest(decomp.evecs);
    if (RTest.det() < 0) {
        std::swap(Devals[0], Devals[1]);
        for (int i=0; i<3; i++) {
            std::swap(decomp.evecs(i,0), decomp.evecs(i,1));
        }
    }
    if (!RTest.isRotationMatrix()) {
        throw std::runtime_error("The transformation to the stretching tensor frame is not a rotation.");
    }

    // D in its principal frame
    hpp::Tensor2<T> DPrincipal(3,3);
    for (int i=0; i<3; i++) {
        DPrincipal(i,i) = Devals[i];
    }

    // Norm out strain rate
    decomp.DNorm = DPrincipal.frobeniusNorm();

    // Components of D_0 decomposition
    std::valarray<T> D0evals = Devals/decomp.DNorm;
    T D0evalsMag = 0.0;
    for (unsigned int i=0; i<D0evals.size(); i++) {
     D0evalsMag += std::pow(D0evals[i],2.0);
    }
    D0evalsMag = std::sqrt(D0evalsMag);
    D0evals /= D0evalsMag;
    T E1 = D0evals[0]/std::sqrt(2.0/3.0);
    T E2 = D0evals[1]/std::sqrt(2.0/3.0);
    T E3 = D0evals[2]/std::sqrt(2.0/3.0);
     
    // Solve for the angle theta
    T theta1 = std::acos(-E3);
    T theta2 = 2.0*M_PI - std::acos(-E3);
    T theta3 = std::acos(E1) + M_PI/3.0;
    //T theta4 = 7.0*M_PI/3.0 - std::acos(E1); //Not needed, but here for completeness
    T theta5 = std::acos(E2) - M_PI/3.0;
    T theta6 = 5.0*M_PI/3.0 - std::acos(E2);

    // Get the matching theta
    T theta;
    T equiv = std::numeric_limits<T>::epsilon()*10000.0;
    if (std::abs(theta1-theta3)<equiv) {
        theta = theta1;
    }
    else {
        theta = theta2;
    }

    // Verify
    T minError = std::min(std::abs(theta-theta5), std::abs(theta-theta6));
    if (minError > equiv) {
        std::string errorMsg = std::string("Could not correctly determine theta. Error = ");
        errorMsg += std::to_string(minError);
        throw std::runtime_error(errorMsg);
    }

    // Return
    decomp.theta = theta;
    return decomp;
}

/**
 * @brief Construct a fourth order isotropic elasticity tensor
 * @param mu the elastic shear modulus \f$\mu\f$
 * @param kappa the elastic bulk modulus \f$\kappa\f$
 * @return The elasticity tensor
 */
template <typename U>
hpp::Tensor4<U> isotropicElasticityTensor(U mu, U kappa) {
    hpp::Tensor4<U> L(3,3,3,3);
    L = 2*mu*hpp::identityTensor4<U>(3) ;
    L += ((U)(kappa-(2.0/3)*mu))*outer<U>(identityTensor2<U>(3), identityTensor2<U>(3));
    return L;
}

/**
 * @brief Construct a fourth order elasticity tensor accounting for cubic symmetry
 * @param c11 elastic constant \f${C}_11\f$
 * @param c12 elastic constant \f${C}_12\f$
 * @param c44 elastic constant \f${C}_44\f$
 * @return The elasticity tensor
 */
template <typename U>
hpp::Tensor4<U> cubeSymmetricElasticityTensor(U c11, U c12, U c44) {
    hpp::Tensor4<U> L(3,3,3,3);  
    U cbar = c11-c12-2*c44;
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {
                for (int l=0; l<3; l++) {
                    L(i,j,k,l) = c12*(i==j)*(k==l);
                    L(i,j,k,l) += c44*((i==k)*(j==l)+(i==l)*(j==k));
                    for (int r=0; r<3; r++) {
                        L(i,j,k,l) += cbar*(i==r)*(j==r)*(k==r)*(l==r);
                    }
                }
            }
        }
    }
    return L;
}

} //END NAMESPACE HPP
    
#endif /* HPP_CONTINUUM_H */