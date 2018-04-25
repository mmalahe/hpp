/// @file rotation.h
/// @author Michael Malahe
/// @brief Header file for rotation classes and functions
#ifndef HPP_ROTATION_H
#define HPP_ROTATION_H

#include "mpi.h"
#include <hpp/config.h>
#include <hpp/tensor.h>
#include <hpp/mpiUtils.h>
#include <hpp/external/ISOI/grid_generation.h>
#include <algorithm>

namespace hpp
{

// Types of symmetry in Schoenflies notation
enum SymmetryType {
    SYMMETRY_TYPE_NONE,
    SYMMETRY_TYPE_C4
};
    
/**
 * @class EulerAngles
 * @author Michael Malahe
 * @date 28/08/17
 * @file rotation.h
 * @brief Euler Angles
 * @details Defined according to the following conventions:
 * - Right-handed
 * - Counter-clockwise
 * - Active
 * - Extrinsic
 * - Z-X-Z sequence
 * - \f$\alpha\f$: The second applied Z rotation angle, \f$[0,2\pi) \f$
 * - \f$\beta\f$: The X rotation angle, \f$[0,\pi) \f$
 * - \f$\gamma\f$: The first applied Z rotation angle, \f$[0,2\pi) \f$
 * - The resulting rotation matrix is the product 
 * \f$ R = Z(\alpha) X(\beta) Z(\gamma) \f$,
 * Where Z is the elemental rotation matrix about z, and X is the elemental
 * rotation matrix about X. 
 */
template <typename T>
struct EulerAngles {
    T alpha = 0;
    T beta = 0;
    T gamma = 0;
    
    HPP_NVCC_ONLY(__host__ __device__) EulerAngles() {;}
    HPP_NVCC_ONLY(__host__ __device__) EulerAngles(T alpha, T beta, T gamma) :
    alpha(alpha), beta(beta), gamma(gamma) {;}
    
    // Getters/setters (mainly intended for Python interface)
    T getAlpha() const {return alpha;}
    T getBeta() const {return beta;}
    T getGamma() const {return gamma;}
    void setAlpha(const T& alpha) {this->alpha = alpha;}
    void setBeta(const T& beta) {this->beta = beta;}
    void setGamma(const T& gamma) {this->gamma = gamma;}
};

template <typename T>
MPI_Datatype getEulerAnglesTypeMPI() {
    MPI_Datatype dtype;
    MPI_Type_contiguous(3, MPIType<T>(), &dtype);
    MPI_Type_commit(&dtype);
    return dtype;
}

/**
 * @brief Convert polar angles to Euler angles
 * @param theta the azimuthal angle
 * @param phi the zenithal angle
 * @return The Euler angles
 */
template <typename T>
EulerAngles<T> polarToEuler(T theta, T phi) {
    EulerAngles<T> angles;
    // Rotate about z axis until x axis is aligned with y axis
    angles.gamma = M_PI/2;
    
    // Rotate about x axis to get the zenithal angle correct
    angles.beta = M_PI/2-phi;
    
    // Rotate about z axis to get the azimuthal angle correct
    angles.alpha = theta;
    
    // Return
    return angles;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const EulerAngles<T>& angles)
{
    out << "[";
    out << angles.alpha << ",";
    out << angles.beta << ",";
    out << angles.gamma;
    out << "]";
    return out;
}

template <typename T>
bool operator==(const EulerAngles<T>& l, const EulerAngles<T>& r) {
    if (l.alpha != r.alpha) return false;
    if (l.beta != r.beta) return false;
    if (l.gamma != r.gamma) return false;
    
    // All checks passed
    return true;
}

template <typename T>
bool operator!=(const EulerAngles<T>& l, const EulerAngles<T>& r) {
    return !(l==r);
}

template <typename T>
Tensor2<T> toRotationMatrix(T alpha, T beta, T gamma) {
    Tensor2<T> R(3,3);
    T c1 = std::cos(alpha);
    T c2 = std::cos(beta);
    T c3 = std::cos(gamma);
    T s1 = std::sin(alpha);
    T s2 = std::sin(beta);
    T s3 = std::sin(gamma);
    R(0,0) = c1*c3 - c2*s1*s3;
    R(0,1) = -c1*s3 - c2*c3*s1;
    R(0,2) = s1*s2;
    R(1,0) = c3*s1 + c1*c2*s3;
    R(1,1) = c1*c2*c3 - s1*s3;
    R(1,2) = -c1*s2;
    R(2,0) = s2*s3;
    R(2,1) = c3*s2;
    R(2,2) = c2;
    return R;
}

template <typename T>
Tensor2<T> toRotationMatrix(EulerAngles<T> angle) {
    return toRotationMatrix(angle.alpha, angle.beta, angle.gamma);
}

/**
 * @brief Get Euler angles from rotation matrix
 * @param R the rotation matrix
 * @return the angles
 */
template <typename T>
EulerAngles<T> toEulerAngles(Tensor2<T> R)
{      
    EulerAngles<T> angle;
    
    // Angle beta
    angle.beta = std::acos(R(2,2));
    
    if (angle.beta > 1e3*std::numeric_limits<T>::epsilon()) {
        // The other 2 angles
        angle.alpha = std::atan2(R(0,2),-R(1,2));
        angle.gamma = std::atan2(R(2,0),R(2,1));
    }
    else {
        // Singular case
        angle.beta = 0.0;
        T alphaPlusGamma = std::atan2(-R(0,1), R(0,0));
        
        // Not uniquely determined, so just pick a combination
        angle.alpha = alphaPlusGamma/2.0;
        angle.gamma =  alphaPlusGamma/2.0;
    }
    
    // Correct the angle ranges if necessary
    if (angle.alpha < 0) angle.alpha += 2*M_PI;
    if (angle.gamma < 0) angle.gamma += 2*M_PI;
    
    // Return
    return angle;
}

/**
 * @brief Convert from quaternion to rotation matrix
 * @param q
 * @return 
 */
template <typename T>
Tensor2<T> toRotationMatrix(const isoi::Quaternion& q) {
    Tensor2<T> R(3,3);
    auto q1 = q.a;
    auto q2 = q.b;
    auto q3 = q.c;
    auto q4 = q.d;
    R(0,0) = 1.0 - 2.0*std::pow(q3,2.0) - 2.0*std::pow(q4,2.0);
    R(0,1) = 2.0*q2*q3 - 2.0*q4*q1;
    R(0,2) = 2.0*q2*q4 + 2.0*q3*q1;
    R(1,0) = 2.0*q2*q3 + 2.0*q4*q1;
    R(1,1) = 1.0 - 2.0*std::pow(q2,2.0) - 2.0*std::pow(q4,2.0);
    R(1,2) = 2.0*q3*q4 - 2.0*q2*q1;
    R(2,0) = 2.0*q2*q4 - 2.0*q3*q1;
    R(2,1) = 2.0*q3*q4 + 2.0*q2*q1;
    R(2,2) = 1.0 - 2.0*std::pow(q2,2.0) - 2.0*std::pow(q3,2.0);
    return R;
}

/**
 * @brief Convert from rotation matrix to quaternion
 * @param R
 * @return 
 */
template <typename T>
EulerAngles<T> toQuaternion(const Tensor2<T>& R) {
    auto t = R.tr();
    isoi::Quaternion q;
    if (t > 0) {
        auto r = std::sqrt((T)1.0 + t);
        auto s = (T)1.0/((T)2.0*r);
        q.a = (T)0.5*r;
        q.b = (R(2,1) - R(1,2))*s;
        q.c = (R(0,2) - R(2,0))*s;
        q.d = (R(1,0) - R(0,1))*s;
    }
    else {
        // sqrtArg must be the largest diagonal element minus the other two
        std::vector<T> diag = {R(0,0), R(1,1), R(2,2)};
        auto diagArgmax = std::distance(diag.begin(), std::max_element(diag.begin(), diag.end()));
        T sqrtArg = diag[diagArgmax];
        for (int i=0; i<3; i++) {
            if (i != diagArgmax) sqrtArg -= diag[i];
        }
        auto r = std::sqrt(sqrtArg);
        auto s = (T)1.0/((T)2.0*r);
        q.a = (T)0.5*r;
        q.b = (R(2,1) - R(1,2))*s;
        q.c = (R(0,2) - R(2,0))*s;
        q.d = (R(1,0) - R(0,1))*s;
        switch (diagArgmax) {
            case 0:
                std::swap(q.a, q.b);
                break;
            case 1:
                std::swap(q.a, q.c);
                break;
            case 2:
                std::swap(q.a, q.d);
                break;
            default:
                throw std::runtime_error("Argmax of the diagonal should be either 0, 1 or 2.");
        }
    }
    
    // Return
    return q;    
}

template <typename T>
EulerAngles<T> toEulerAngles(const isoi::Quaternion& q) {
    return toEulerAngles(toRotationMatrix<T>(q));
}

template <typename T>
isoi::Quaternion toQuaternion(const EulerAngles<T>& angles) {
    return toQuaternion(toRotationMatrix(angles));
}

//////////////////////
// RANDOM ROTATIONS //
//////////////////////

// TRANSFORMATIONS
inline std::mt19937 makeMt19937(bool defaultSeed) {
    if (defaultSeed) {
        return std::mt19937();
    }
    else {
        std::random_device rd;
        return std::mt19937(rd());
    }
}

// Arvo 1992 method
template <typename T>
void rotationTensorFrom3UniformRandomsArvo1992(Tensor2<T>& A, T x1, T x2, T x3) {    
    // Static to avoid expensive alloc/dealloc for the tens/hundreds of millions regime
    static std::vector<T> v(3);
    static Tensor2<T> R(3,3);
    static Tensor2<T> H(3,3);
    
    // Random rotation about vertical axis
    R(0,0) = std::cos(2*M_PI*x1);
    R(1,1) = R(0,0);
    R(0,1) = std::sin(2*M_PI*x1);
    R(1,0) = -R(0,1);
    R(2,2) = 1.0;
    
    // Random rotation of vertical axis    
    v[0] = std::cos(2*M_PI*x2)*std::sqrt(x3);
    v[1] = std::sin(2*M_PI*x2)*std::sqrt(x3);
    v[2] = std::sqrt(1-x3);
    
    // Householder reflection
    outerInPlace(v, v, H);
    H *= (T)2.0;
    H(0,0) -= 1.0;
    H(1,1) -= 1.0;
    H(2,2) -= 1.0;
    
    // Final matrix
    A = H*R;
}

template <typename T>
void randomRotationTensorArvo1992(std::mt19937& gen, Tensor2<T>& A) {    
    // Static to avoid expensive alloc/dealloc for the tens/hundreds of millions regime
    static std::vector<T> v(3);
    static Tensor2<T> R(3,3);
    static Tensor2<T> H(3,3);
    
    // double is specified here, as if it is not, the values out of the
    // distribution are different for float and double, even with the same
    // default seed
    static std::uniform_real_distribution<double> dist(0.0,1.0);
    
    // Random samples
    T x1 = (T)dist(gen);
    T x2 = (T)dist(gen);
    T x3 = (T)dist(gen);
    
    // Generate 
    rotationTensorFrom3UniformRandomsArvo1992(A, x1, x2, x3);
}

template <typename T>
void randomRotationTensorArvo1992(Tensor2<T>& A, bool defaultSeed=false) {
    // Create generator once
    static std::mt19937 gen = makeMt19937(defaultSeed);
    
    // Generate
    randomRotationTensorArvo1992(gen, A);
}

template <typename T>
void rotationTensorFrom3UniformRandoms(Tensor2<T>& A, T x1, T x2, T x3) {
    rotationTensorFrom3UniformRandomsArvo1992(A, x1, x2, x3);
} 

/**
 * @brief Generate a random rotation tensor.
 * @details This is translated from scipy/linalg/tests/test_decomp.py
 * The description there, verbatim is:
 * Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(n)).
    The algorithm is described in the paper
    Stewart, G.W., 'The efficient generation of random orthogonal
    matrices with an application to condition estimators', SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization
 * @param dim
 * @param defaultSeed if true, use a default random seed, otherwise generate a truly random one
 * @return The rotation tensor
 */
template <typename T>
Tensor2<T> randomRotationTensorStewart1980(unsigned int dim, bool defaultSeed=false) {
    // Generator
    static std::mt19937 gen = makeMt19937(defaultSeed);
    
    // Normal distribution to draw from
    std::normal_distribution<T> dist(0.0,1.0);   
    
    // Construction
    Tensor2<T> H = identityTensor2<T>(dim);
    std::vector<T> D = ones<T>(dim);
    for (unsigned int n=1; n<dim; n++) {
        std::valarray<T> x(dim-n+1);
        for (unsigned int i=0; i<dim-n+1; i++) {
            x[i] = dist(gen);
        }
        D[n-1] = std::copysign(1.0, x[0]);
        x[0] -= D[n-1]*std::sqrt((x*x).sum());
        
        // Householder transformation
        Tensor2<T> Hx = identityTensor2<T>(dim-n+1) - (T)2.0*outer(x, x)/(x*x).sum();
        Tensor2<T> mat = identityTensor2<T>(dim);
        for (unsigned int i=n-1; i<dim; i++) {
            for (unsigned int j=n-1; j<dim; j++) {
                mat(i,j) = Hx(i-n+1,j-n+1);
            }
        }
        H = H*mat;
    }
    
    // Fix the last sign such that the determinant is 1
    D[dim-1] = -prod(D);
    H = (D*H.trans()).trans();
    
    // Return
    return H;
}

/**
 * @brief Generate a random rotation tensor
 * @param dim
 * @param A the output tensor to populate
 * @param defaultSeed if true, use a default random seed, otherwise generate a truly random one
 */
template <typename T>
void randomRotationTensorInPlace(unsigned int dim, Tensor2<T>& A, bool defaultSeed=false) {
    // Generate tensor
    if (dim == 3) {
        randomRotationTensorArvo1992<T>(A, defaultSeed);
    }
    else {
        A = randomRotationTensorStewart1980<T>(dim, defaultSeed);
    }
    
    // Check the tensor
    #ifdef DEBUG_BUILD
        if (!(A.isRotationMatrix())) {
            std::cerr << "Warning: random rotation tensor didn't pass check. Re-generating." << std::endl;
            randomRotationTensorInPlace<T>(dim, A, defaultSeed);
        }
    #endif
}

template <typename T>
Tensor2<T> randomRotationTensor(unsigned int dim, bool defaultSeed=false) {
    // Generate tensor
    Tensor2<T> A(dim,dim);
    randomRotationTensorInPlace<T>(dim, A, defaultSeed);
    return A;
}

/////////////////////
// ROTATION SPACES //
/////////////////////

/**
 * @class SO3Discrete
 * @author Michael Malahe
 * @date 27/03/18
 * @file rotation.h
 * @brief A discrete and uniform sampling of SO(3)
 * @detail Using the methods and implementations from the following work:
 * Generating Uniform Incremental Grids on SO(3) Using the Hopf Fibration.
   Anna Yershova, Swati Jain, Steven M. LaValle, and Julie C. Mitchell,
   International Journal of Robotics Research, IJRR 2009 
 */
template <typename T>
class SO3Discrete {
public:
    SO3Discrete() {;}
    SO3Discrete(unsigned int resolution, SymmetryType symmetryType = SYMMETRY_TYPE_NONE);
    isoi::Quaternion getQuat(unsigned int i) {return quatList[i];}
    EulerAngles<T> getEulerAngle(unsigned int i) {return eulerAngleList[i];}
    unsigned int size() {return quatList.size();}
private:
    SymmetryType symmetryType;
    std::vector<isoi::Quaternion> quatList;
    std::vector<EulerAngles<T>> eulerAngleList;
};

/**
 * @brief Calculate the minimum orientation space resolution that will use at least the number of points given.
 * @param nPointsInt
 * @param symmetryType
 * @return The resolution
 * @todo Add and test accounting for C4 symmetry
 */
inline int orientationSpaceResolutionRequiredForNumberOfPoints(long int nPointsInt, SymmetryType symmetryType = SYMMETRY_TYPE_NONE) {
    if (symmetryType == SYMMETRY_TYPE_C4) {
        std::cerr << "WARNING: C4 symmetry implementation is not complete." << std::endl;
        std::cerr << "Proceeding without accounting for symmetries." << std::endl;
        symmetryType = SYMMETRY_TYPE_NONE;
    }
    double nPoints = nPointsInt;
    double nPointsSymm;
    switch (symmetryType) {
        case SYMMETRY_TYPE_NONE:
            nPointsSymm = nPoints;
            break;
        case SYMMETRY_TYPE_C4:
            nPointsSymm = nPoints/4;
            break;
        default:
            throw std::runtime_error("Don't know about this symmetry type.");
    }
    double res = std::log(nPointsSymm*72)/std::log(8.0);
    int resInt = std::ceil(res);
    return resInt;
}

} //END NAMESPACE HPP

#endif /* HPP_ROTATION_H */