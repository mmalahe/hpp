/** @file spectralUtils.cu
* @author Michael Malahe
* @brief Source file for spectral utilities CUDA implementations
* @detail
*/

#include <hpp/config.h>
#include <hpp/spectralUtilsCUDA.h>

namespace hpp
{
#ifdef HPP_USE_CUDA

template <typename T, unsigned int N>
SpectralDatabaseCUDA<T,N>::SpectralDatabaseCUDA(){
    ;
}

template <typename T, unsigned int N>
SpectralDatabaseCUDA<T,N>::SpectralDatabaseCUDA(const SpectralDatabase<T>& dbIn, const std::vector<SpectralDatasetID>& dsetIDs) {    
    // Number of datasets
    nDsets = dsetIDs.size();
    
    // Only take in power of two grid dimensions
    unsigned int gridLength = dbIn.getGridDims()[0];
    if (gridLength != pow(2,log2u(gridLength))) {
        throw std::runtime_error("Grid length is not a power of two.");
    }
    
    // Grid dimension conversions
    gridDims = makeDeviceCopyVec(dbIn.getGridDims());
    gridDimsSharedPtr = cudaSharedPtr(gridDims);
    
    std::vector<T> gridStartsCorrectType(dbIn.getGridStarts().begin(),dbIn.getGridStarts().end());
    gridStarts = makeDeviceCopyVec(gridStartsCorrectType);
    gridStartsSharedPtr = cudaSharedPtr(gridStarts);
    
    std::vector<T> gridStepsCorrectType(dbIn.getGridSteps().begin(),dbIn.getGridSteps().end());
    gridSteps = makeDeviceCopyVec(gridStepsCorrectType);
    gridStepsSharedPtr = cudaSharedPtr(gridSteps);
    
    // Dsets is a pointer on the host
    // It points to device memory
    // That device memory contains other pointers to device memory
    // Allocate enough device memory to contain the top-level pointers
    dsets = allocDeviceMemory<SpectralDatasetCUDA<T,N>>(nDsets);
    dsetsSharedPtr = cudaSharedPtr(dsets);
    for (unsigned int i=0; i<nDsets; i++) {
        SpectralDataset<T> dsetRawH = dbIn.getDataset(dsetIDs[i]);
        
        // Check compatability
        if (dsetRawH.getNDims() != N) {
            throw std::runtime_error("Mismatch in dataset dimensions.");
        }
        
        // Correct data format on the host
        unsigned int nTerms = dsetRawH.getNTerms();
        this->nTermsTypical = nTerms;
        std::vector<SpectralCoordCUDA<N>> coordsH(nTerms);
        std::vector<SpectralCoeffCUDA<T>> coeffsH(nTerms);
        for (unsigned int j=0; j<nTerms; j++) {            
            // Copy spectral coordinates
            int *coordsStart = dsetRawH.getCoords().get() + N*j;
            std::copy(coordsStart, coordsStart+N, &(coordsH[j](0)));
            
            // Copy complex coefficient
            T *coeffStart = dsetRawH.getCoeffs().get()+2*j;
            coeffsH[j].re = *coeffStart;
            // The imaginary part is pre-negated to save an operation later when
            // computing the IDFT. Amounts to about a 7% saving.
            coeffsH[j].im = -*(coeffStart+1);
        }
        
        // Create the dataset on the host, with "data" pointing to device memory
        SpectralDatasetCUDA<T,N> dsetD;
        dsetD.nTerms = nTerms;
        dsetD.coords = makeDeviceCopyVec(coordsH);
        dsetD.coeffs = makeDeviceCopyVec(coeffsH);      
        coordSharedPtrs.push_back(cudaSharedPtr(dsetD.coords));
        coeffSharedPtrs.push_back(cudaSharedPtr(dsetD.coeffs));
        
        // Copy the dataset to the device
        CUDA_CHK(cudaMemcpy(dsets+i, &dsetD, sizeof(SpectralDatasetCUDA<T,N>), cudaMemcpyHostToDevice));        
    }
}

// Host function
template <typename T, unsigned int N>
T SpectralDatabaseCUDA<T,N>::getIDFTRealH(unsigned int dsetIdx, std::vector<unsigned int> spatialCoord) const {    
    // Device memory for result
    std::shared_ptr<T> valD = allocDeviceMemorySharedPtr<T>();
    std::shared_ptr<unsigned int> spatialCoordD = makeDeviceCopyVecSharedPtr(spatialCoord);
    
    // Run the kernel
    std::shared_ptr<SpectralDatabaseCUDA<T,N>> thisD = makeDeviceCopySharedPtr(*this);
    GET_IDFT_REAL<<<1,1>>>(thisD.get(), dsetIdx, spatialCoordD.get(), valD.get());
    CUDA_CHK(cudaDeviceSynchronize());

    // Get value back
    T valH = getHostValue(valD);
    return valH;
}

///////////////////////////////
// SPECTRAL DATABASE UNIFIED //
///////////////////////////////

template <typename T, unsigned int N, unsigned int P>
SpectralDatabaseUnifiedCUDA<T,N,P>::SpectralDatabaseUnifiedCUDA(){
    ;
}

template <typename T, unsigned int N, unsigned int P>
SpectralDatabaseUnifiedCUDA<T,N,P>::SpectralDatabaseUnifiedCUDA(const SpectralDatabaseUnified<T>& dbIn, const std::vector<SpectralDatasetID>& dsetIDs) {    
    // Number of datasets
    nDsets = dsetIDs.size();
    
    // Check compatability
    if (nDsets != P) {
        throw std::runtime_error("Wrong number of datasets.");
    }
    
    // Only take in power of two grid dimensions
    unsigned int gridLength = dbIn.getGridDims()[0];
    if (gridLength != pow(2,log2u(gridLength))) {
        throw std::runtime_error("Grid length is not a power of two.");
    }
    
    // Grid dimension conversions
    gridDims = makeDeviceCopyVec(dbIn.getGridDims());
    gridDimsSharedPtr = cudaSharedPtr(gridDims);
    
    std::vector<T> gridStartsCorrectType(dbIn.getGridStarts().begin(),dbIn.getGridStarts().end());
    gridStarts = makeDeviceCopyVec(gridStartsCorrectType);
    gridStartsSharedPtr = cudaSharedPtr(gridStarts);
    
    std::vector<T> gridStepsCorrectType(dbIn.getGridSteps().begin(),dbIn.getGridSteps().end());
    gridSteps = makeDeviceCopyVec(gridStepsCorrectType);
    gridStepsSharedPtr = cudaSharedPtr(gridSteps);
    
    // Read in coordinates
    nTerms = dbIn.getNTerms();
    std::vector<SpectralDataUnifiedCUDA<T,N,P>> dataH(nTerms);
    for (unsigned int i=0; i<nTerms; i++) {            
        // Copy spectral coordinates
        int *coordsStart = dbIn.getCoords().get() + N*i;
        std::copy(coordsStart, coordsStart+N, &(dataH[i].coord(0)));
    }
    
    // Read in data
    for (unsigned int j=0; j<nDsets; j++) { 
        // Dataset ID
        SpectralDatasetID dsetID = dsetIDs[j];
        
        // Correct data format on the host        
        for (unsigned int i=0; i<nTerms; i++) {
            // Copy complex coefficient
            T *coeffStart = dbIn.getCoeffs(dsetID).get()+2*i;
            dataH[i].coeffs[j].re = *coeffStart;
            // The imaginary part is pre-negated to save an operation later when
            // computing the IDFT. Amounts to about a 7% saving.
            dataH[i].coeffs[j].im = -*(coeffStart+1);
        }      
    }
    
    // Copy data to device
    data = makeDeviceCopyVec(dataH);    
    dataSharedPtr = cudaSharedPtr(data);
}

// Explicit instantiations of spectral database
template class SpectralDatabaseCUDA<float, 2>;
template class SpectralDatabaseCUDA<double, 2>;
template class SpectralDatabaseCUDA<float, 4>;
template class SpectralDatabaseCUDA<double, 4>;
template class SpectralDatabaseUnifiedCUDA<float, 4, 9>;
template class SpectralDatabaseUnifiedCUDA<double, 4, 9>;

#endif /* HPP_USE_CUDA */
}//END NAMESPACE HPP