/** @file testCrystal.cu
* @author Michael Malahe
* @brief Tests for functions in crystalCUDA.h
*/

#include <hpp/config.h>
HPP_CHECK_CUDA_ENABLED_BUILD
#include <hpp/tensor.h>
#include <hpp/gshCUDA.h>
#include <stdexcept>
#include <iostream>
#include <limits>

namespace hpp{
    
template <typename T>
void testGSHCUDA() 
{
    // In/out test
    int nGSHLevels = 5;
    int nReals = nGSHReals(nGSHLevels);
    std::vector<T> gshRealsIn(nReals);
    gshRealsIn[0] = 1.0;
    gshRealsIn[1] = 0.0;
    for (int i=2; i<nReals; i++) {
        gshRealsIn[i] = i; 
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

} //END NAMESPACE HPP

int main(int argc, char *argv[]) 
{
    // Test 
    hpp::testGSHCUDA<float>();
    hpp::testGSHCUDA<double>(); 
    
    // Return
    return 0;
}