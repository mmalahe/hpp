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
Tensor2<U> stretchingVelocityGradient(U theta, U eDot) {
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
    Tensor2<U> D_0(3,3);
    for (int i=0; i<3; i++) {
        std::vector<U> e_i = basis[i];
        D_0 += D_comps[i]*outer(e_i, e_i);
    }
    Tensor2<U> D = eDot*D_0;
    
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
StretchingTensorDecomposition<T> getStretchingTensorDecomposition(const Tensor2<T>& L) {
    // The decomposition
    StretchingTensorDecomposition<T> decomp;

    // Stretching tensor from velocity gradient
    Tensor2<T> D = ((T)0.5)*(L + L.trans());

    // Evec decomposition of D
    std::valarray<T> Devals;
    D.evecDecomposition(Devals, decomp.evecs);
    
    // Ensure that the determinant of the eigenvector matrix is positive
    // for the purposes of treating it like a rotation
    if (decomp.evecs.det() < 0) {
        std::swap(Devals[0], Devals[1]);
        for (int i=0; i<3; i++) {
            std::swap(decomp.evecs(i,0), decomp.evecs(i,1));
        }
    }
    if (!decomp.evecs.isRotationMatrix()) {
        Tensor2<T> product = decomp.evecs*decomp.evecs.trans();
        std::cerr << "R*R^T = " << product << std::endl;
        throw std::runtime_error("The transformation to the stretching tensor frame is not a rotation.");
    }

    // D in its principal frame
    Tensor2<T> DPrincipal(3,3);
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
Tensor4<U> isotropicElasticityTensor(U mu, U kappa) {
    Tensor4<U> L(3,3,3,3);
    L = 2*mu*identityTensor4<U>(3) ;
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
Tensor4<U> cubeSymmetricElasticityTensor(U c11, U c12, U c44) {
    Tensor4<U> L(3,3,3,3);  
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

/**
 * @brief Volume average fourth order tensors
 * @param tVec a vector of the tensors
 * @param vVec a vector of the volumes in which each tensor holds
 * @return the volume average
 */
template <typename U>
Tensor4<U> volumeAverage(const std::vector<Tensor4<U>>& tVec, const std::vector<U>& vVec) {
    // Check sizes
    unsigned int nT = tVec.size();
    unsigned int nV = vVec.size();
    if (nT != nV) {
        throw std::runtime_error("Different number of stiffness tensors and volumes.");
    }
    
    // Sum up
    Tensor4<U> tTot = tVec[0];
    U vTot = vVec[0];
    for (unsigned int i=1; i<nT; i++) {
        tTot += tVec[i];
        vTot += vVec[i];
    }
    
    // Average
    Tensor4<U> tAvg = tTot/vTot;
    return tAvg;
}

template <typename U>
Tensor4<U> getEshelbyTensorCubicMaterialSphericalInclusion(U c11, U c12, U c44, U I0, U I1, U I2) {
    U mu = c11 - c12 - 2.0*c44;
    U a = mu*(c11+c12)/(c11*c44);
    U b = std::pow(mu, 2.0)*(c11+2*c12+c44)/(c11*std::pow(c44,2.0));
    U m = c11-c44;
    U p = c12+c44;
    
    // Construct E
    Tensor4<U> A0(3,3,3,3);
    Tensor4<U> A1(3,3,3,3);
    Tensor4<U> A2(3,3,3,3);
    A0(0,0,0,0) = (1.0/3.0)*c11;
    A1(0,0,0,0) = (2*m/3.0)*c11*c44;
    A2(0,0,0,0) = a/c44;
    A0(0,0,1,1) = 0.0;
    A1(0,0,1,1) = -(p/3.0)*c11*c44;
    A2(0,0,1,1) = -p*mu/(c11*std::pow(c44,2.0));
    A0(0,1,0,1) = (1.0/6.0)*c44;
    A1(0,1,0,1) = a*(mu-2.0*c44)/(12.0*mu*c44);
    A2(0,1,0,1) = (1.0/2.0)*(a/(2.0*c44)-b/mu);   
    Tensor4<U> E = I0*A0+I1*A1+I2*A2;
    
    
    
    // Note that S has the both minor symmetries S_{ijkl} = S{jikl} = S{ijlk}
    // Here we apply leading minor symmetry to E before using it to construct S
    E(1,0,0,1) = E(0,1,0,1);
    
    // EXPERIMENT
    E(0,1,1,0) = E(0,1,0,1);
    
    std::cout << "E = " << E << std::endl;
    
    // Construct S
    Tensor4<U> C = cubeSymmetricElasticityTensor(c11, c12, c44);
    
    std::cout << "C = " << C << std::endl;
    
    Tensor4<U> S = contract(C, E);
    
    // Return
    return S;
}

/**
 * @brief Get the homogenized stiffness tensor using a volume average.
 * @param cVec a vector of stiffness tensors to homogenize
 * @param vVec a vector of volumes occupied by the stiffness tensors
 * @return the homogenized stiffness tensor
 */
template <typename U>
Tensor4<U> getHomogenizedStiffnessVolumeAverage(const std::vector<Tensor4<U>>& cVec, const std::vector<U>& vVec) {
    // Check sizes
    unsigned int nC = cVec.size();
    unsigned int nV = vVec.size();
    if (nC != nV) {
        throw std::runtime_error("Different number of stiffness tensors and volumes.");
    }
    
    // Get volume average
    Tensor4<U> cTot = (U)0.0*cVec[0];
    U vTot = 0.0;
    for (unsigned int i=0; i<cVec.size(); i++) {
        cTot += cVec[i]*vVec[i];
        vTot += vVec[i];
    }
    Tensor4<U> cBar = cTot/vTot;
    return cBar;
}

/**
 * @brief Get the homogenized stiffness tensor using the elastic self-consistent method.
 * @param cVec a vector of stiffness tensors to homogenize
 * @param vVec a vector of volumes occupied by the stiffness tensors
 * @param S the Eshelby tensor
 * @return the homogenized stiffness tensor
 */
template <typename U>
Tensor4<U> getHomogenizedStiffnessELSC(const std::vector<Tensor4<U>>& cVec, const std::vector<U>& vVec, const Tensor4<U>& S) {
    // Check sizes
    unsigned int nC = cVec.size();
    unsigned int nV = vVec.size();
    if (nC != nV) {
        throw std::runtime_error("Different number of stiffness tensors and volumes.");
    }
    
    // Convergence criterion
    U cTol = 1.0e2*std::numeric_limits<U>::epsilon();
    int maxIters = 100;
    
    // Initialize
    Tensor4<U> cBar = volumeAverage(cVec, vVec);
    U vTot = 0.0;
    for (const auto& v : vVec) {
        vTot += v;
    }
    
    // Common terms
    Tensor4<U> SInv = S.inv();
    
    // Single allocations for frequent terms
    Tensor4<U> Ar = (U)0.0*cBar;
    Tensor4<U> cBarTot = (U)0.0*cBar;
    Tensor4<U> cBarPrev = (U)0.0*cBar;
    Tensor4<U> cTilde = (U)0.0*cBar;
    Tensor4<U> cDiff = (U)0.0*cBar;
    
    // Iterate
    bool converged = false;
    for (unsigned int i=0; i<maxIters; i++) {
        // Previous
        cBarPrev = cBar;
        
        // Common terms
        cTilde = contract(contract(cBarPrev, SInv), identityTensor4<U>(3)-S);
        
        // Volume average of ELSC update terms
        cBarTot = (U)0.0*cBar;
        for (unsigned int r=0; r<nC; r++) {
            Ar = contract((cVec[r]+cTilde).inv(), cBarPrev+cTilde);
            cBarTot += contract(cVec[r], Ar)*vVec[r];
        }
        cBar = cBarTot/vTot;
        
        // Check convergence
        cDiff = cBar-cBarPrev;
        U relDiff = cDiff.frobeniusNorm()/cBar.frobeniusNorm();
        if (relDiff < cTol) {
            converged = true;
            break;
        }
    }
    
    if (!converged) {
        throw std::runtime_error("ELSC homogenization failed to converge.");
    }

    // Return
    return cBar;
}

} //END NAMESPACE HPP
    
#endif /* HPP_CONTINUUM_H */