/** @file testCrystal.cu
* @author Michael Malahe
* @brief Tests for functions in crystalCUDA.h
*/

#include <hpp/config.h>
HPP_CHECK_CUDA_ENABLED_BUILD
#include <hpp/crystalCUDA.h>
#include <hpp/casesUtils.h>
#include <iostream>

namespace hpp{
    
template <typename T>
void testSpectralPolycrystalGSHCUDA() 
{
    // Crystal properties and database
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
    auto polycrystalGSH = SpectralPolycrystalGSHCUDA<T, crystalType>(props, db, init.s_0);
    polycrystalGSH.resetUniformRandomOrientations(init.s_0);
    auto ncrystals = polycrystalGSH.getNumRepresentativeCrystals();
    std::cout << "Comparison being done with " << ncrystals << " crystals" << std::endl;
    auto polycrystalSpectral = SpectralPolycrystalCUDA<T, nSlipSystems(crystalType)>(ncrystals, init.s_0, props, db);
    
    // Applied deformation
    T t = 0.0;
    T strainRate = 1e-3;
    auto L = planeStrainCompressionVelocityGradient(t, strainRate);
    T dt = 20;
    
    // Take steps
    for (int i=0; i<50; i++) {
        polycrystalSpectral.step(L, dt);
        polycrystalGSH.step(L, dt);
    }
    
    // Compare
    auto gshSpectral = polycrystalSpectral.getGSHCoeffs();
    auto gshGSH = polycrystalGSH.getGSHCoeffs();
    std::cout << "SpectralPolycrystalCUDA" << std::endl;
    std::cout << gshSpectral;
    std::cout << "SpectralPolycrystalGSHCUDA" << std::endl;
    std::cout << gshGSH;
}

} //END NAMESPACE HPP

int main(int argc, char *argv[]) 
{
    // Test 
    hpp::testSpectralPolycrystalGSHCUDA<float>(); 
    
    // Return
    return 0;
}