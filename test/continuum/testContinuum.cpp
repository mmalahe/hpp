/** @file testContinuum.cpp
* @author Michael Malahe
* @brief Tests for functions in continuum.h.
*/

#include <stdexcept>
#include <limits>
#include <string>
#include <random>

#include <hpp/continuum.h>

namespace hpp
{

template <typename U>
void testStretchingTensorDecomposition() {
    // Equivalence threshold
    const U closeEnough = 10000*std::numeric_limits<U>::epsilon();
    
    // Test basic stretching tensor construction
    U DSimpleNorm = 2.0;
    Tensor2<U> DSimple = stretchingVelocityGradient<U>(0, DSimpleNorm);
    Tensor2<U> DCorrect(3,3);
    DCorrect(0,0) = DSimpleNorm*std::sqrt(2.0/3.0)*0.5;
    DCorrect(1,1) = DSimpleNorm*std::sqrt(2.0/3.0)*0.5;
    DCorrect(2,2) = -DSimpleNorm*std::sqrt(2.0/3.0)*1.0;
    if (DSimple != DCorrect) {
        throw std::runtime_error("Incorrect stretching velocity gradient.");
    }
    
    // Test stretching tensor decomposition
    int nRandomTheta = 5;
    int nRandomOrientations = 5;
    std::random_device rd;
    std::mt19937 randomgen(rd());
    std::uniform_real_distribution<double> dist(0.0,2*M_PI);
    for (int iTheta=0; iTheta<nRandomTheta; iTheta++) {
        U theta = dist(randomgen);
        for (int iOrientation=0; iOrientation<nRandomOrientations; iOrientation++) {
            Tensor2<U> R = randomRotationTensor<U>(3);
            U DNorm = 2.0;    
            Tensor2<U> DPrincipal = stretchingVelocityGradient(theta, DNorm);
            Tensor2<U> D = R*DPrincipal*R.trans();
            StretchingTensorDecomposition<U> decomp = getStretchingTensorDecomposition(D);
            Tensor2<U> DBackOutPrincipal = stretchingVelocityGradient(decomp.theta, decomp.DNorm);
            Tensor2<U> DBackOut = decomp.evecs*DBackOutPrincipal*decomp.evecs.trans();   
            std::cout << "D = " << D << std::endl;
            std::cout << "DBackOut = " << DBackOut << std::endl;
            U error = (D-DBackOut).frobeniusNorm()/D.frobeniusNorm();
            if (error > closeEnough) {
                std::string errorMsg("D != DBackOut. Error = ");
                errorMsg += std::to_string(error);
                throw std::runtime_error(errorMsg);
            }
            
            // Check norm
            if (std::abs(DNorm-decomp.DNorm)/std::abs(DNorm) > closeEnough) {
                std::string errMsg = "Mismatch in DNorm. Input = "+std::to_string(DNorm);
                errMsg += ", Output = "+std::to_string(decomp.DNorm);
                throw std::runtime_error(errMsg);
            }        
        }
    }
}    

template <typename U>
void testHomogenizations() {
    // Equivalence threshold
    const U closeEnough = 100*std::numeric_limits<U>::epsilon();
    
    // Using material constants from Kneer1965, page 831
    U c11 = 169.05;
    U c12 = 121.93;
    U c44 = 75.5;
    U I0 = 1.858;
    U I1 = 0.4101;
    U I2 = 0.02072;
    
    // Volume average
    Tensor4<U> c1 = cubeSymmetricElasticityTensor<U>(1.0*c11, 1.0*c12, 1.0*c44);
    Tensor4<U> c2 = cubeSymmetricElasticityTensor<U>(2.0*c11, 2.0*c12, 2.0*c44);
    U v1 = 3.0;
    U v2 = 1.0;
    Tensor4<U> cBarAnalyticVolumeAverage = cubeSymmetricElasticityTensor<U>(1.25*c11, 1.25*c12, 1.25*c44);
    std::vector<Tensor4<U>> cVec = {c1, c2};
    std::vector<U> vVec = {v1, v2};
    Tensor4<U> cBarVolumeAverage = getHomogenizedStiffnessVolumeAverage(cVec, vVec); 
    U error = (cBarVolumeAverage-cBarAnalyticVolumeAverage).frobeniusNorm()/cBarAnalyticVolumeAverage.frobeniusNorm();
    if (error > closeEnough) {
        throw std::runtime_error("Mismatch in volume averages.");
    }
}

template <typename U>
void testContinuum() 
{
    testStretchingTensorDecomposition<U>();
    testHomogenizations<U>();
}    
    
}//END NAMESPACE HPP


int main(int argc, char *argv[]) 
{    
    // Test
    hpp::testContinuum<float>();
    hpp::testContinuum<double>();  

    // Return
    return 0;
}