/** @file casesUtils.h
* @author Michael Malahe
* @brief Header file for quantities common to multiple application cases
* @details
*/

#ifndef HPP_CASESUTILS_H
#define HPP_CASESUTILS_H

#include <hpp/config.h>
#include <hpp/tensor.h>
#include <functional>
#include <stdexcept>

namespace hpp 
{

// Simple shear experiment
template <typename U>
hpp::Tensor2<U> simpleShearDeformationGradient(U t, U shear_rate) {
    hpp::Tensor2<U> F = hpp::identityTensor2<U>(3);
    F(0,1) = shear_rate*t;
    return F;
}

template <typename U>
hpp::Tensor2<U> simpleShearVelocityGradient(U t, U shear_rate) {
    hpp::Tensor2<U> L(3,3);
    L(0,1) = shear_rate;
    return L;
}

// Simple compression experiment
template <typename U>
hpp::Tensor2<U> simpleCompressionDeformationGradient(U t, U comp_rate) {
    hpp::Tensor2<U> F(3,3);
    F(0,0) = std::exp(-0.5*comp_rate*t);
    F(1,1) = std::exp(-0.5*comp_rate*t);
    F(2,2) = std::exp(1.0*comp_rate*t);
    return F;
}

template <typename U>
hpp::Tensor2<U> simpleCompressionVelocityGradient(U t, U comp_rate) {
    hpp::Tensor2<U> L(3,3);
    L(0,0) = -0.5*comp_rate;
    L(1,1) = -0.5*comp_rate;
    L(2,2) = 1.0*comp_rate;
    return L;
}

// Plane strain compression experiment
template <typename U>
hpp::Tensor2<U> planeStrainCompressionDeformationGradient(U t, U comp_rate) {
    hpp::Tensor2<U> F(3,3);
    F(0,0) = std::exp(1.0*comp_rate*t);
    F(1,1) = 1.0;
    F(2,2) = std::exp(-1.0*comp_rate*t);
    return F;
}

template <typename U>
hpp::Tensor2<U> planeStrainCompressionVelocityGradient(U t, U comp_rate) {
    hpp::Tensor2<U> L(3,3);
    L(0,0) = 1.0*comp_rate;
    L(2,2) = -1.0*comp_rate;
    return L;
}

// Null experiment
template <typename U>
hpp::Tensor2<U> staticDeformationGradient(U t) {
    hpp::Tensor2<U> F = hpp::identityTensor2<U>(3);
    return F;
}

template <typename U>
hpp::Tensor2<U> staticVelocityGradient(U t) {
    hpp::Tensor2<U> L(3,3);
    return L;
}

template <typename U>
class Experiment {
    
    public:
        Experiment(std::string experimentName);
        
        // Members
        U tStart;
        U tEnd;
        U strainRate;
        std::function<hpp::Tensor2<U>(U)> F_of_t;
        std::function<hpp::Tensor2<U>(U)> L_of_t;
};

} // END NAMESPACE hpp

#endif /* HPP_CASESUTILS_H */