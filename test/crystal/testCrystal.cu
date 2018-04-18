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
void testPolycrystalGSHCUDA() 
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
    
    // Polycrystal
    auto polycrystal = SpectralPolycrystalGSHCUDA<T, crystalType>(props, db, init.s_0);
    polycrystal.resetUniformRandomOrientations(init.s_0);
    
    // Applied deformation
    T t = 0.0;
    T strainRate = 1e-3;
    auto L = planeStrainCompressionVelocityGradient(t, strainRate);
    T dt = 20;
    
    // Take a step
    polycrystal.step(L, dt);
    
    // Get output state
    auto gshOut = polycrystal.getGSHCoeffs();
}

} //END NAMESPACE HPP

int main(int argc, char *argv[]) 
{
    // Test 
    hpp::testPolycrystalGSHCUDA<float>(); 
    
    // Return
    return 0;
}