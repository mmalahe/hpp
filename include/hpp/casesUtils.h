/** @file casesUtils.h
* @author Michael Malahe
* @brief Header file for quantities common to multiple application cases
* @details
*/

#ifndef HPP_CASESUTILS_H
#define HPP_CASESUTILS_H

#include <hpp/config.h>
#include <hpp/tensor.h>
#include <hpp/rotation.h>
#include <functional>
#include <stdexcept>
#include <memory>

namespace hpp 
{

/**
 * @brief The deformation gradient from a simple shear.
 * @details The deformation gradient is the identity with the addition of
 * \f$ \mathbf{F}(0,1) = \dot{\varepsilon} t \f$.
 * @param t The current time \f$t\f$
 * @param shear_rate The shear rate \f$ \dot{\varepsilon}\f$
 * @tparam U the scalar type
 * @return \f$ \mathbf{F} \f$
 */
template <typename U>
Tensor2<U> simpleShearDeformationGradient(U t, U shear_rate) {
    Tensor2<U> F = identityTensor2<U>(3);
    F(0,1) = shear_rate*t;
    return F;
}

/**
 * @brief The velocity gradient from a simple shear.
 * @details The velocity gradient is zero with the addition of
 * \f$ \mathbf{L}(0,1) = \dot{\varepsilon} \f$.
 * @param t The current time \f$t\f$
 * @param shear_rate The shear rate \f$ \dot{\varepsilon}\f$
 * @tparam U the scalar type
 * @return \f$ \mathbf{L} \f$
 */
template <typename U>
Tensor2<U> simpleShearVelocityGradient(U t, U shear_rate) {
    Tensor2<U> L(3,3);
    L(0,1) = shear_rate;
    return L;
}

/**
 * @brief The deformation gradient from a simple compression.
 * @details The deformation gradient is zero with the addition of
 * \f$\mathbf{F}(0,0) = \exp(-0.5\dot{\varepsilon} t)\f$, 
 * \f$\mathbf{F}(1,1) = \exp(-0.5\dot{\varepsilon} t)\f$,
 * \f$\mathbf{F}(2,2) = \exp(1.0\dot{\varepsilon} t)\f$.
 * @param t The current time \f$t\f$
 * @param comp_rate the compression rate \f$ \dot{\varepsilon}\f$
 * @tparam U the scalar type
 * @return \f$ \mathbf{F} \f$
 */
template <typename U>
Tensor2<U> simpleCompressionDeformationGradient(U t, U comp_rate) {
    Tensor2<U> F(3,3);
    F(0,0) = std::exp(-0.5*comp_rate*t);
    F(1,1) = std::exp(-0.5*comp_rate*t);
    F(2,2) = std::exp(1.0*comp_rate*t);
    return F;
}

/**
 * @brief The velocity gradient from a simple compression.
 * @details The velocity gradient is zero with the addition of
 * \f$\mathbf{L}(0,0) = -0.5\dot{\varepsilon}\f$, 
 * \f$\mathbf{L}(1,1) = -0.5\dot{\varepsilon}\f$,
 * \f$\mathbf{L}(2,2) = 1.0\dot{\varepsilon}\f$.
 * @param t The current time \f$t\f$
 * @param comp_rate the compression rate \f$ \dot{\varepsilon}\f$
 * @tparam U the scalar type
 * @return \f$ \mathbf{L} \f$
 */
template <typename U>
Tensor2<U> simpleCompressionVelocityGradient(U t, U comp_rate) {
    Tensor2<U> L(3,3);
    L(0,0) = -0.5*comp_rate;
    L(1,1) = -0.5*comp_rate;
    L(2,2) = 1.0*comp_rate;
    return L;
}

/**
 * @brief The deformation gradient from a plane strain compression.
 * @details The deformation gradient is zero with the addition of
 * \f$\mathbf{F}(0,0) = \exp(1.0\dot{\varepsilon} t)\f$, 
 * \f$\mathbf{F}(1,1) = 1.0\f$,
 * \f$\mathbf{F}(2,2) = \exp(-1.0\dot{\varepsilon} t)\f$.
 * @param t The current time \f$t\f$
 * @param comp_rate
 * @tparam U the scalar type
 * @return \f$ \mathbf{F} \f$
 */
template <typename U>
Tensor2<U> planeStrainCompressionDeformationGradient(U t, U comp_rate) {
    Tensor2<U> F(3,3);
    F(0,0) = std::exp(1.0*comp_rate*t);
    F(1,1) = 1.0;
    F(2,2) = std::exp(-1.0*comp_rate*t);
    return F;
}

/**
 * @brief The velocity gradient from a plane strain compression.
 * @details The velocity gradient is zero with the addition of
 * \f$\mathbf{L}(0,0) = 1.0\dot{\varepsilon}\f$, 
 * \f$\mathbf{L}(2,2) = -1.0\dot{\varepsilon}\f$.
 * @param t The current time \f$t\f$
 * @param comp_rate the compression rate \f$ \dot{\varepsilon}\f$
 * @tparam U the scalar type
 * @return \f$ \mathbf{L} \f$
 */
template <typename U>
Tensor2<U> planeStrainCompressionVelocityGradient(U t, U comp_rate) {
    Tensor2<U> L(3,3);
    L(0,0) = 1.0*comp_rate;
    L(2,2) = -1.0*comp_rate;
    return L;
}

/**
 * @brief The deformation gradient for no deformation.
 * @details The deformation gradient is the identity.
 * @param t The current time \f$t\f$
 * @tparam U the scalar type
 * @return \f$ \mathbf{F} \f$
 */
template <typename U>
Tensor2<U> staticDeformationGradient(U t) {
    Tensor2<U> F = identityTensor2<U>(3);
    return F;
}

/**
 * @brief The velocity gradient for no deformation.
 * @details The velocity gradient is zero.
 * @param t The current time \f$t\f$
 * @tparam U the scalar type
 * @return \f$ \mathbf{L} \f$
 */
template <typename U>
Tensor2<U> staticVelocityGradient(U t) {
    Tensor2<U> L(3,3);
    return L;
}

/**
 * @class OrientationGenerator
 * @author Michael Malahe
 * @brief An abstract class for reproducibly generating a sequence of orientations.
 * @date 22/09/17
 * @file casesUtils.h
 * @tparam U the scalar type
 */
template <typename U>
class OrientationGenerator {
    public:
        /**
         * @brief Generate the next orientation in the sequence.
         * @param rotMatrix the matrix into which to write the new rotation matrix
         * @tparam U the scalar type
         */
        virtual void generateNext(Tensor2<U>& rotMatrix) = 0;
        /**
         * @brief Generate the next orientation in the sequence.
         * @param angles the set of angles into which to write the new angles
         * @tparam U the scalar type
         */
        virtual void generateNext(EulerAngles<U>& angles) {
            Tensor2<U> rotMatrix(3,3);
            this->generateNext(rotMatrix);
            angles = toEulerAngles(rotMatrix);
        }
        /**
         * @brief Default destructor
         */
        virtual ~OrientationGenerator() {};
};

/**
 * @class RandomOrientationGenerator
 * @author Michael Malahe
 * @brief A class for reproducibly generating a sequence of random orientations.
 * @date 03/10/17
 * @file casesUtils.h
 * @tparam U the scalar type
 */
template <typename U>
class RandomOrientationGenerator : public OrientationGenerator<U> {
    public:
        /**
         * @brief Default constructor
         */
        RandomOrientationGenerator() {;}
        /**
         * @brief Generate the next orientation in the sequence.
         * @param rotMatrix the matrix into which to write the new rotation matrix
         * @tparam U the scalar type
         */
        virtual void generateNext(Tensor2<U>& rotMatrix) {
            randomRotationTensorInPlace(3, rotMatrix, true);
        } 
};

/**
 * @class GridOrientationGenerator
 * @author Michael Malahe
 * @brief Generates orientations on a fixed grid over an azimuthal angle of
 * \f$ [0, 2\pi) \f$ and a zenithal angle of \f$ [0, \pi/2) \f$. Does not cover
 * the area of a sphere equally.
 * @date 28/09/17
 * @file casesUtils.h
 * @tparam U the scalar type
 */
template <typename U>
class GridOrientationGenerator : public OrientationGenerator<U> {
    public:
        /**
         * @brief Default constructor
         * @param nTheta number of grid points in the azimuthal angle
         * @param nPhi number of grid points in the zenithal angle
         */
        GridOrientationGenerator(int nTheta=8, int nPhi=4);
        /**
         * @brief Generate the next orientation in the sequence.
         * @param angles the set of angles into which to write the new angles
         * @tparam U the scalar type
         */
        virtual void generateNext(EulerAngles<U>& angles) {
            // Get angles
            U theta = iTheta*dTheta;
            U phi = iPhi*dPhi;
            angles = polarToEuler(theta, phi);
            
            // Prepare next angle
            iPhi++;
            if (iPhi == nPhi) {
                iPhi = 0;
                iTheta++;
                iTheta = iTheta % nTheta;
            }
        };
        /**
         * @brief Generate the next orientation in the sequence.
         * @param rotMatrix the matrix into which to write the new rotation matrix
         * @tparam U the scalar type
         */
        virtual void generateNext(Tensor2<U>& rotMatrix) {
            EulerAngles<U> angles;
            this->generateNext(angles);
            rotMatrix = toRotationMatrix(angles);
        };
        
    private:        
        int nTheta; ///< the number of grid points for the azimuthal angle
        int nPhi; ///< the number of grid points for the zenithal angle
        int iTheta = 0; ///< the azimuthal index of the orientation to generate next
        int iPhi = 0; ///< the zenithal index of the orientation to generate next
        U dTheta; ///< the spacing between azimuthal angles on the grid
        U dPhi; ///< the spacing between zenithal angles on the grid
};

/**
 * @class Experiment
 * @author Michael Malahe
 * @date 03/10/17
 * @file casesUtils.h
 * @brief A class to represent the conditions of an imposed strain experiment.
 */
template <typename U>
class Experiment {
    
    public:
        /**
         * @brief Construct from a hard-coded experiment name.
         * @param experimentName the name of the experiment
         */
        explicit Experiment(std::string experimentName);
        U tStart; ///< the start time of the experiment
        U tEnd; ///< the end time of the experiment
        U strainRate; ///< the strain rate throughout the experiment
        /// the function defining the deformation gradient for a given strain
        /// rate and time into the experiment
        std::function<Tensor2<U>(U)> F_of_t;
        /// the function defining the velocity gradient for a given strain
        /// rate and time into the experiment
        std::function<Tensor2<U>(U)> L_of_t;
        /// the generator of crystal orientations for the experiment
        std::shared_ptr<OrientationGenerator<U>> orientationGenerator;
        
        // Generating next orientation
        void generateNextOrientationInPlace(Tensor2<U>& rotMatrix) {this->orientationGenerator->generateNext(rotMatrix);}
        void generateNextOrientationInPlace(EulerAngles<U>& angles) {this->orientationGenerator->generateNext(angles);}
        EulerAngles<U> generateNextOrientationAngles() {
            EulerAngles<U> angles;
            this->orientationGenerator->generateNext(angles);
            return angles;
        }
        
        // Getters/setters (mainly intended for Python interface)
        U getTStart() const {return tStart;}
        U getTEnd() const {return tEnd;}
        U getStrainRate() const {return strainRate;}
        std::function<Tensor2<U>(U)> getF_of_t() const {return F_of_t;}
        std::function<Tensor2<U>(U)> getL_of_t() const {return L_of_t;}
};

} // END NAMESPACE hpp

#endif /* HPP_CASESUTILS_H */