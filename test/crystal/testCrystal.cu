/** @file testCrystal.cu
* @author Michael Malahe
* @brief Tests for functions in crystalCUDA.h
*/

#include <hpp/config.h>
HPP_CHECK_CUDA_ENABLED_BUILD
#include <hpp/crystalCUDA.h>
#include <hpp/casesUtils.h>
#include <limits>
#include <iostream>

namespace hpp{

/**
 * @brief Checks if the internal conversion from gsh->orientations->gsh is the identity
 */
template <typename T, hpp::CrystalType CRYSTAL_TYPE>
void testPolycrystalGSHCUDAConversionIdentity(SpectralPolycrystalGSHCUDA<T, CRYSTAL_TYPE>& polycrystal, const GSHCoeffsCUDA<T>& gshIn) {
    polycrystal.setOrientations(gshIn);
    auto gshOut = polycrystal.getGSHCoeffs();
    auto relErr = (gshOut-gshIn).norm()/gshIn.norm();
    T tol = 5000.0*std::numeric_limits<T>::epsilon();
    if (relErr > tol) {
        std::cerr << "testPolycrystalGSHCUDAConversionIdentity" << std::endl;
        std::cerr << "GSH In = " << std::endl;
        std::cerr << gshIn << std::endl;
        std::cerr << "GSH Out = " << std::endl;
        std::cerr << gshOut << std::endl;
        auto densities = polycrystal.getDensities();
        std::cerr << "Min density = " << *std::min_element(densities.begin(), densities.end()) << std::endl;
        std::cerr << "Max density = " << *std::max_element(densities.begin(), densities.end()) << std::endl;
        std::cerr << "Density sum = " << polycrystal.getDensitySum() << std::endl;
        std::cerr << "Relative error in GSH = " << relErr << std::endl;
        std::cerr << "Tolerance = " << tol << std::endl;
        throw std::runtime_error("Relative error too high.");
    }
}    

/**
 * @brief
 * @todo Add test where at each timestep, GSH polycrystal is fed its own GSH
 * coefficients for a reset, so that that is the only state that is maintained.
 */
template <typename T>
void testSpectralPolycrystalGSHCUDA() 
{
    // Crystal properties
    CrystalInitialConditions<T> init = defaultCrystalInitialConditions<T>();
    const CrystalType crystalType = CRYSTAL_TYPE_FCC;
    CrystalPropertiesCUDA<T, nSlipSystems(crystalType)> props(defaultCrystalProperties<T>());
    
    // Crystal database
    std::string dbFilename = "../../cases/databases/voce/databaseSpectralOrderedUnified128.hdf5";
    std::vector<SpectralDatasetID> dsetIDs = defaultCrystalSpectralDatasetIDs();
    int nTerms = 8192;
    int refinementMultiplier = 128;    
    SpectralDatabaseUnified<T> db(dbFilename, dsetIDs, nTerms, refinementMultiplier);
    
    // Polycrystals
    unsigned int orientationSpaceResolution = 5;
    auto polycrystalGSH = SpectralPolycrystalGSHCUDA<T, crystalType>(props, db, init.s_0, orientationSpaceResolution);
    auto ncrystals = polycrystalGSH.getNumRepresentativeCrystals();
    std::cout << "Comparison being done with " << ncrystals << " crystals" << std::endl;
    auto polycrystalSpectral = SpectralPolycrystalCUDA<T, nSlipSystems(crystalType)>(ncrystals, init.s_0, props, db);
    
    // Single-class tests
    GSHCoeffsCUDA<T> gshHardcoded;
    gshHardcoded.set(0,0,0,make_cuComplex((T)1.0,(T)0.0));
    gshHardcoded.set(1,-1,1,make_cuComplex((T)1.23,(T)0.0));
    gshHardcoded.set(2,1,-1,make_cuComplex((T)-4.56,(T)0.0));
    gshHardcoded.set(3,0,0,make_cuComplex((T)7.89,(T)0.0));
    gshHardcoded.set(4,0,0,make_cuComplex((T)1.01,(T)0.0));
    testPolycrystalGSHCUDAConversionIdentity<T, crystalType>(polycrystalGSH, gshHardcoded);    
    auto gshUniform = uniformOrientationGSHCoeffsCUDA<T>();
    testPolycrystalGSHCUDAConversionIdentity<T, crystalType>(polycrystalGSH, gshUniform);    
    
    // Prepare for comparison tests
    unsigned long int randomSeed = 0;
    polycrystalSpectral.setToInitialConditionsRandomOrientations(init.s_0, randomSeed);
    polycrystalGSH.setOrientations(polycrystalSpectral.getGSHCoeffs());
    
    // Applied deformation
    T t = 0.0;
    T strainRate = 1e-3;
    auto L = planeStrainCompressionVelocityGradient(t, strainRate);
    T dt = 20;
    
    // Take steps
    for (int i=0; i<50; i++) {
        polycrystalSpectral.step(L, dt);
        //polycrystalGSH.setOrientations(polycrystalGSH.getGSHCoeffs());
        polycrystalGSH.step(L, dt);
    }
    
    // Compare
    auto gshSpectral = polycrystalSpectral.getGSHCoeffs();
    auto gshGSH = polycrystalGSH.getGSHCoeffs();
    auto relErr = (gshGSH-gshSpectral).norm()/gshSpectral.norm();
    T tol = 110000.0*std::numeric_limits<T>::epsilon();
    if (relErr > tol) {
        std::cerr << "testSpectralPolycrystalGSHCUDA" << std::endl;
        std::cerr << "GSH Spectral = " << std::endl;
        std::cerr << gshSpectral << std::endl;
        std::cerr << "GSH GSH = " << std::endl;
        std::cerr << gshGSH << std::endl;
        auto densities = polycrystalGSH.getDensities();
        std::cerr << "Min density = " << *std::min_element(densities.begin(), densities.end()) << std::endl;
        std::cerr << "Max density = " << *std::max_element(densities.begin(), densities.end()) << std::endl;
        std::cerr << "Relative error in GSH = " << relErr << std::endl;
        std::cerr << "Tolerance = " << tol << std::endl;
        throw std::runtime_error("Relative error too high.");
    }
}

} //END NAMESPACE HPP

int main(int argc, char *argv[]) 
{
    // Test 
    hpp::testSpectralPolycrystalGSHCUDA<float>(); 
    
    // Return
    return 0;
}