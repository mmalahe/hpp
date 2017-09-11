/** @file testSpectralUtils.cpp
* @author Michael Malahe
* @brief Tests for spectral utilities functions
*/

#include <iostream>
#include <limits>
#include <stdexcept>
#include <hpp/config.h>
#include <hpp/spectralUtils.h>
#include <hpp/spectralUtilsCUDA.h>

namespace hpp
{

template <typename U>
struct SpectralDataRawDataPair {
    SpectralDatabase<U> spectral;
    std::vector<U> raw;
};
    
/**
 * @brief Generate a spectral database and corresponding raw data for testing purposes.
 * @detail 
 * @return 
 */
template <typename U>
SpectralDataRawDataPair<U> generateTestData1() {
    // Simple 4x4 grid
    unsigned int nDims = 2;
    unsigned int L = 4;
    std::vector<unsigned int> gridDims = {L, L};
    std::vector<double> gridStarts = {0.0, 0.0};
    std::vector<double> gridEnds = {2*M_PI, 2*M_PI};
    unsigned int nGridPoints = std::accumulate(gridDims.begin(), gridDims.end(), 1, std::multiplies<unsigned int>());
    
    // Derived grid info
    std::vector<U> gridLengths(nDims);
    std::vector<U> gridSteps(nDims);
    for (unsigned int i=0; i<nDims; i++) {
        gridLengths[i] = gridEnds[i] - gridStarts[i];
        gridSteps[i] = gridLengths[i]/gridDims[i];
    }
    
    // Create raw data
    std::vector<U> rawData(nGridPoints);
    for (unsigned int flatIdx=0; flatIdx<nGridPoints; flatIdx++) {
        std::vector<unsigned int> idx = unflatC(flatIdx, gridDims);
        U x = idx[0]*gridSteps[0];
        U y = idx[1]*gridSteps[1];
        rawData[flatIdx] = std::cos(x) + std::sin(y);
    }
    
    // Spectral data
    std::vector<std::vector<unsigned int>> coordsList;
    std::vector<std::complex<U>> coeffList;
    
    // Add coordinates and coefficient for cos term
    {
        std::vector<unsigned int> coord = {1,0};
        std::complex<U> coeff(1.0, 0.0);
        coordsList.push_back(coord);
        coeffList.push_back(coeff);
    }
    
    // Add coordinates and coefficient for sin term
    {
        std::vector<unsigned int> coord = {0,1};
        std::complex<U> coeff(0.0, -1.0);
        coordsList.push_back(coord);
        coeffList.push_back(coeff);       
    }
    
    // Create spectral dataset
    SpectralDataset<U> dset(coordsList, coeffList);
    
    // Create spectral database
    SpectralDatasetID dsetID("test");
    std::map<SpectralDatasetID, SpectralDataset<U>> dsets;
    dsets[dsetID] = dset;
    SpectralDatabase<U> dbase(gridDims, gridStarts, gridEnds, dsets);

    // Store data
    SpectralDataRawDataPair<U> dataPair;
    dataPair.raw = rawData;
    dataPair.spectral = dbase;
    
    // Return
    return dataPair;
}

template <typename U>
void testSpectralDatabaseCUDA() {
    // Get the test data
    SpectralDataRawDataPair<U> dataPair = generateTestData1<U>();
    
    // Set up grid dimensions
    std::vector<unsigned int> gridDims = dataPair.spectral.getGridDims();
    unsigned int nGridPoints = std::accumulate(gridDims.begin(), gridDims.end(), 1, std::multiplies<unsigned int>());
    
    // Convert host spectral database to CUDA one
    SpectralDatasetID dsetID("test");
    std::vector<SpectralDatasetID> dsetIDs = {dsetID};
    SpectralDatabaseCUDA<U, 2> dbaseCUDA(dataPair.spectral, dsetIDs);
    
    // Test the quality of dataset agreement
    U closeEnough = 10*std::numeric_limits<U>::epsilon();
    std::vector<U> errors(nGridPoints);
    for (unsigned int flatIdx=0; flatIdx<nGridPoints; flatIdx++) {
        std::vector<unsigned int> idx = unflatC(flatIdx, gridDims);
        U rawData = dataPair.raw[flatIdx];        
        U spectralData = dbaseCUDA.getIDFTRealH(0, idx);
        U absoluteError = std::abs((rawData-spectralData));
        if (absoluteError > closeEnough) {
            std::cout << flatIdx << " " << rawData << " " << spectralData << std::endl;
            std::cerr << "Absolute error: " << absoluteError << std::endl;
            throw std::runtime_error("Absolute error is too high.");
        }
    }
}

} //END NAMESPACE HPP

int main(int argc, char *argv[]) {    
    hpp::testSpectralDatabaseCUDA<float>();
    hpp::testSpectralDatabaseCUDA<double>();
    
    // Return
    return 0;
}



