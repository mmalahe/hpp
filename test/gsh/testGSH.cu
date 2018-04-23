/** @file testCrystal.cu
* @author Michael Malahe
* @brief Tests for functions in crystalCUDA.h
*/

#include <hpp/config.h>
HPP_CHECK_CUDA_ENABLED_BUILD
#include <hpp/tensor.h>
#include <hpp/rotation.h>
#include <hpp/crystalCUDA.h>
#include <hpp/gshCUDA.h>
#include <hpp/rotation.h>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace hpp{

template <typename T>
std::vector<SpectralCrystalCUDA<T>> generateUniformlyOrientedCrystals(unsigned int orientationSpaceResolution = 2) {
    SO3Discrete<T> orientationSpace(orientationSpaceResolution);
    std::vector<SpectralCrystalCUDA<T>> crystals(orientationSpace.size());
    T initS = 0.0; //immaterial what this is, we only care about the angles
    for (unsigned int i=0; i<orientationSpace.size(); i++) {
        crystals[i].s = initS;
        crystals[i].angles = orientationSpace.getEulerAngle(i);
    }
    return crystals;
}

template <typename T>
std::vector<SpectralCrystalCUDA<T>> generateRandomlyOrientedCrystals(unsigned int ncrystals) {
    std::vector<SpectralCrystalCUDA<T>> crystals(ncrystals);
    T initS = 0.0; //immaterial what this is, we only care about the angles
    auto R = randomRotationTensor<T>(3);
    for (unsigned int i=0; i<ncrystals; i++) {
        randomRotationTensorInPlace(3, R);
        crystals[i].s = initS;
        crystals[i].angles = toEulerAngles(R);
    }
    return crystals;
}

template <typename T>
void testGSHCUDAVectorInOut() {
    // In/out test from std::vector
    int nGSHLevels = 5;
    int nReals = nGSHReals(nGSHLevels);
    std::vector<T> gshRealsIn(nReals);
    gshRealsIn[0] = 1.0;
    gshRealsIn[1] = 0.0;
    for (int i=2; i<nReals; i++) {
        gshRealsIn[i] = 0;
    }
    GSHCoeffsCUDA<T> gsh(gshRealsIn, nGSHLevels);
    auto gshRealsOut = gsh.getReals(nGSHLevels);
    if (gshRealsIn.size() != gshRealsOut.size()) {
        std::cerr << "GSH size in = " << gshRealsIn.size() << std::endl;
        std::cerr << "GSH size out = " << gshRealsOut.size() << std::endl;
        throw std::runtime_error("Mismatch in gsh sizes.");
    }
    T err = 0.0;
    for (int i=0; i<nReals; i++) {
        err += std::pow(gshRealsOut[i]-gshRealsIn[i], (T)2.0);
    }
    err = std::sqrt(err)/nReals;
    T closeEnough = 100.0*std::numeric_limits<T>::epsilon();
    if (err > closeEnough) {
        std::cerr << "Error in GSH in/out = " << err << std::endl;
        throw std::runtime_error("Error in GSH in/out is too high");
    }
}

template <typename T>
void testGSHCUDAUniformOrientation() {
    auto crystals = generateUniformlyOrientedCrystals<T>(2);
    auto gsh = getGSHFromCrystalOrientations(crystals);
    T tol = 5e-3;
    if (!gsh.isUniform(tol)) {
        std::cerr << "Coefficients = " << std::endl;
        std::cerr << gsh << std::endl;
        std::cerr << "Off uniform mass = " << std::endl;
        std::cerr << gsh.offUniformMass() << std::endl;
        throw std::runtime_error("Coefficients in non-uniform components are too high for this to be regarded as a uniform distribution of orientations.");
    }
    
}

template <typename T>
void testGSHCUDARandomOrientation() {
    auto crystals = generateRandomlyOrientedCrystals<T>(std::pow(2, 15));
    auto gsh = getGSHFromCrystalOrientations(crystals);
    T tol = 1e-1;
    if (!gsh.isUniform(tol)) {
        std::cerr << "Coefficients = " << std::endl;
        std::cerr << gsh << std::endl;
        std::cerr << "Off uniform mass = " << std::endl;
        std::cerr << gsh.offUniformMass() << std::endl;
        throw std::runtime_error("Coefficients in non-uniform components are too high for this to be regarded as a random distribution of orientations.");
    }
}

template <typename T>
void testGSHCUDA() 
{
    testGSHCUDAVectorInOut<T>();
    testGSHCUDAUniformOrientation<T>();
    testGSHCUDARandomOrientation<T>(); 
}

} //END NAMESPACE HPP

int main(int argc, char *argv[]) 
{
    // Test 
    hpp::testGSHCUDA<float>();
    hpp::testGSHCUDA<double>(); 
    
    // Return
    return 0;
}