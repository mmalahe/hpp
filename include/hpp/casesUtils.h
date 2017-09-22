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
#include <memory>

namespace hpp 
{

// Simple shear experiment
template <typename U>
Tensor2<U> simpleShearDeformationGradient(U t, U shear_rate) {
    Tensor2<U> F = identityTensor2<U>(3);
    F(0,1) = shear_rate*t;
    return F;
}

template <typename U>
Tensor2<U> simpleShearVelocityGradient(U t, U shear_rate) {
    Tensor2<U> L(3,3);
    L(0,1) = shear_rate;
    return L;
}

// Simple compression experiment
template <typename U>
Tensor2<U> simpleCompressionDeformationGradient(U t, U comp_rate) {
    Tensor2<U> F(3,3);
    F(0,0) = std::exp(-0.5*comp_rate*t);
    F(1,1) = std::exp(-0.5*comp_rate*t);
    F(2,2) = std::exp(1.0*comp_rate*t);
    return F;
}

template <typename U>
Tensor2<U> simpleCompressionVelocityGradient(U t, U comp_rate) {
    Tensor2<U> L(3,3);
    L(0,0) = -0.5*comp_rate;
    L(1,1) = -0.5*comp_rate;
    L(2,2) = 1.0*comp_rate;
    return L;
}

// Plane strain compression experiment
template <typename U>
Tensor2<U> planeStrainCompressionDeformationGradient(U t, U comp_rate) {
    Tensor2<U> F(3,3);
    F(0,0) = std::exp(1.0*comp_rate*t);
    F(1,1) = 1.0;
    F(2,2) = std::exp(-1.0*comp_rate*t);
    return F;
}

template <typename U>
Tensor2<U> planeStrainCompressionVelocityGradient(U t, U comp_rate) {
    Tensor2<U> L(3,3);
    L(0,0) = 1.0*comp_rate;
    L(2,2) = -1.0*comp_rate;
    return L;
}

// Null experiment
template <typename U>
Tensor2<U> staticDeformationGradient(U t) {
    Tensor2<U> F = identityTensor2<U>(3);
    return F;
}

template <typename U>
Tensor2<U> staticVelocityGradient(U t) {
    Tensor2<U> L(3,3);
    return L;
}

/**
 * @class OrientationGenerator
 * @author Michael Malahe
 * @date 22/09/17
 * @file casesUtils.h
 * @brief An abstract class for reproducibly generating a sequence of orientations.
 */
template <typename U>
class OrientationGenerator {
    public:
        virtual void generateNext(Tensor2<U>& rotMatrix) = 0;
        void generateNext(EulerAngles<U>& angles) {};
        virtual ~OrientationGenerator() {};
};

template <typename U>
class RandomOrientationGenerator : public OrientationGenerator<U> {
    public:
        RandomOrientationGenerator() {;}
        virtual void generateNext(Tensor2<U>& rotMatrix) {rotMatrix(0,0) = 9;}
        
    private:
        
};

template <typename U>
class Experiment {
    
    public:
        explicit Experiment(std::string experimentName);
        
        // Members
        U tStart;
        U tEnd;
        U strainRate;
        std::function<Tensor2<U>(U)> F_of_t;
        std::function<Tensor2<U>(U)> L_of_t;
        std::shared_ptr<OrientationGenerator<U>> orientationGenerator;
};

} // END NAMESPACE hpp

#endif /* HPP_CASESUTILS_H */