#include <hpp/spectralUtils.h>

namespace hpp
{

// INTERFACE WITH FFTW
    
FFTWConfigRealND prepareFFTWConfigRealND(const std::vector<ptrdiff_t>& realDims, MPI_Comm comm) {
    // Blank configuration
    FFTWConfigRealND cfg;
    
    // MPI
    cfg.comm = comm;
    
    // Dimensions
    cfg.realDims = realDims;
    cfg.gridRank = realDims.size();
    cfg.complexDims = realDims;
    cfg.complexDims.back() = cfg.realDims.back()/2+1;
    cfg.NReal = 1;
    for (auto dim : cfg.realDims) {
        cfg.NReal *= dim;
    }
    cfg.NComplex = 1;
    for (auto dim : cfg.complexDims) {
        cfg.NComplex *= dim;
    }
    cfg.realDimsPadded = cfg.realDims;
    cfg.realDimsPadded.back() = cfg.complexDims.back()*2;
    
    // Allocate values in and out
    cfg.nLocalComplexMem = fftw_mpi_local_size(cfg.gridRank, cfg.complexDims.data(), cfg.comm, &(cfg.localN0), &(cfg.local0Start));
    cfg.nLocalRealPadded = 2*cfg.nLocalComplexMem;
    cfg.in = fftw_alloc_real(cfg.nLocalRealPadded);
    cfg.out = fftw_alloc_complex(cfg.nLocalComplexMem);
    cfg.backin = fftw_alloc_real(cfg.nLocalRealPadded);
    
    // Local dimensions
    cfg.realDimsPaddedLocal = cfg.realDimsPadded;
    cfg.realDimsPaddedLocal[0] = cfg.localN0;
    cfg.realDimsLocal = cfg.realDims;
    cfg.realDimsLocal[0] = cfg.localN0;
    cfg.complexDimsLocal = cfg.complexDims;
    cfg.complexDimsLocal[0] = cfg.localN0;
    cfg.nLocalComplex = 1;
    for (auto&& dim : cfg.complexDimsLocal) {
        cfg.nLocalComplex *= dim;
    }
    cfg.nLocalReal = 1;
    for (auto&& dim : cfg.realDimsLocal) {
        cfg.nLocalReal *= dim;
    }
        
    // Execution plan
    cfg.forwardPlan = fftw_mpi_plan_dft_r2c(cfg.gridRank, cfg.realDims.data(), cfg.in, cfg.out, cfg.comm, FFTW_ESTIMATE);
    cfg.backwardPlan = fftw_mpi_plan_dft_c2r(cfg.gridRank, cfg.realDims.data(), cfg.out, cfg.backin, cfg.comm, FFTW_ESTIMATE);
    
    // Return
    return cfg;
}

void destroyConfigRealND(FFTWConfigRealND& cfg) {
    fftw_destroy_plan(cfg.forwardPlan);
    fftw_destroy_plan(cfg.backwardPlan);
    fftw_free(cfg.in);
    fftw_free(cfg.out);
    fftw_free(cfg.backin);
}

// DATABASES

unsigned int roundUpToMultiple(unsigned int val, unsigned int multiple) {
    if (!val%multiple) {
        val = ((val/multiple)+1)*multiple;
    }
    return val;
}

template <typename U>
SpectralDataset<U>::SpectralDataset(const std::vector<std::vector<unsigned int>>& coordsList, const std::vector<std::complex<U>>& coeffList) {
    // Set number of terms
    nTerms = coordsList.size();
    if (coeffList.size() != nTerms) {
        throw std::runtime_error("Mismatch in size of coords and coeffs.");
    }
    
    // Set dimensions
    nDims = coordsList[0].size();
    
    // Allocate aligned memory for coordinates
    unsigned int alignedSize = roundUpToMultiple(nTerms*nDims*sizeof(int), alignment);
    coords = std::shared_ptr<int>((int*)aligned_alloc(alignment, alignedSize), FreeDelete());
    
    // Allocate aligned memory for coefficients
    alignedSize = roundUpToMultiple(2*nTerms*sizeof(U), alignment);
    coeffs = std::shared_ptr<U>((U*)aligned_alloc(alignment, alignedSize), FreeDelete());
    
    // Populate coordinates and coefficients
    for (unsigned int iTerm=0; iTerm<nTerms; iTerm++) {
        std::copy(coordsList[iTerm].begin(), coordsList[iTerm].end(), coords.get()+nDims*iTerm);
        coeffs.get()[iTerm*2] = std::real(coeffList[iTerm]);
        coeffs.get()[iTerm*2+1] = std::imag(coeffList[iTerm]);
    }
}

// Dataset ID comparison for use with std::map
bool operator<(const SpectralDatasetID& l, const SpectralDatasetID& r) {
    // Order by name if base names are different
    if (l.baseName != r.baseName) {
        return l.baseName < r.baseName;
    }
    // Otherwise order by component index
    else {
        unsigned int nComponents = l.component.size();
        if (nComponents != r.component.size()) {
            throw std::runtime_error("Component dimension mismatch.");
        }
        for (unsigned int i=0; i<nComponents; i++) {
            if (l.component[i] < r.component[i]) {
                return true;
            }
            else if (l.component[i] > r.component[i]) {
                return false;
            }
        }
        // If no components are different, then they are equivalent
        return false;
    }
}

///////////////////////
// SPECTRAL DATABASE //
///////////////////////

/**
 * @brief Load a single component from a dataset
 * @param dbfile
 * @param dsetBasename
 * @param componentIdxUint
 */
template <typename U>
void SpectralDatabase<U>::loadDatasetSingleComponent(HDF5MPIHandler& dbfile, std::string dsetBasename, std::vector<unsigned int> componentIdxUint, unsigned int nTerms, unsigned int refineMult) {
    // Property list for reading in data
    hid_t plist_in = dbfile.getPropertyListTransferIndependent();
    
    // Determine names of datasets within the file
    std::string dsetNameCoords = getCoordsName(dsetBasename);
    std::string dsetNameCoeffs = getCoeffsName(dsetBasename);
    
    // Open datasets
    hid_t dsetCoords = dbfile.getDataset(dsetNameCoords);
    hid_t dsetCoeffs = dbfile.getDataset(dsetNameCoeffs);
    
    // Convert component index
    std::vector<hsize_t> componentIdx(componentIdxUint.begin(), componentIdxUint.end());
    
    // Get dimensions
    std::vector<hsize_t> dsetCoordsDims = getDatasetDims(dsetCoords);
    std::vector<hsize_t> dsetCoeffsDims = getDatasetDims(dsetCoeffs);
    unsigned int nCoeffs = dsetCoordsDims.end()[-2];
    unsigned int nDims = dsetCoordsDims.back();
    if (nCoeffs != dsetCoeffsDims.back()) throw std::runtime_error("Mismatch in dimensions of coeffs and coords.");
    if (nTerms > nCoeffs) throw std::runtime_error("Requested more terms than exist in the dataset.");
    unsigned int nTermsToInclude = std::min(nTerms, nCoeffs);
    
    // Read in raw data
    std::vector<unsigned short int> coordsBuffer(nTermsToInclude*nDims);
    std::vector<hdf_complex_t> coeffsBuffer(nTermsToInclude);
    
    HDFReadWriteParamsC coordsReadParams;
    coordsReadParams.dataOffset = componentIdx;
    coordsReadParams.dataOffset.push_back(0);
    coordsReadParams.dataOffset.push_back(0);
    coordsReadParams.dataCount = std::vector<hsize_t>(componentIdx.size(), 1);
    coordsReadParams.dataCount.push_back(nTermsToInclude);
    coordsReadParams.dataCount.push_back(nDims);
    coordsReadParams.datatype = H5Dget_type(dsetCoords);
    readHDF5SimpleArray(dsetCoords, plist_in, coordsReadParams, coordsBuffer.data());
    
    HDFReadWriteParamsC coeffsReadParams;
    coeffsReadParams.dataOffset = componentIdx;
    coeffsReadParams.dataOffset.push_back(0);
    coeffsReadParams.dataCount = std::vector<hsize_t>(componentIdx.size(), 1);
    coeffsReadParams.dataCount.push_back(nTermsToInclude);
    coeffsReadParams.datatype = H5Dget_type(dsetCoeffs);    
    readHDF5SimpleArray(dsetCoeffs, plist_in, coeffsReadParams, coeffsBuffer.data());
    
    // Convert coords and coeffs
    std::vector<std::vector<unsigned int>> coordsList(nTermsToInclude);
    std::vector<std::complex<U>> coeffList(nTermsToInclude);
    
    // The number of actual numerical terms is less than the raw number requested,
    // because using the fact that the data is real, we can simply double
    // the terms that are supposed to have their conjugates included
    unsigned int nTermsIncluded = 0; //number of analytic terms that have been included
    unsigned int nTermsNumerical = 0; //number of numerical terms required to capture their effect
    for (unsigned int i=0; nTermsIncluded<nTermsToInclude; i++){
        // Coords
        coordsList[i].resize(nDims);
        unsigned short int *start = coordsBuffer.data()+i*nDims;
        unsigned short int *end = start+nDims;//Note that "end" is the address one *after* the final element
        std::copy(start, end, coordsList[i].data());
        
        // Spectral refinement
        for (unsigned int j=0; j<nDims; j++) {
            // Insert zeros in the middle
            if (coordsList[i][j] >= (gridDims[j]/refineMult)/2) {
                coordsList[i][j] += gridDims[j]-(gridDims[j]/refineMult);
            }
        }
        
        // Coeffs
        hdf_complex_t rawVal = coeffsBuffer[i];
        coeffList[i] = std::complex<U>((U)rawVal.r, (U)rawVal.i);
        nTermsIncluded++;
        
        // Add complex conjugates
        bool shouldAddConjugate = true;
        
        // In the case where we're double counting the final dimension
        if (coordsList[i].back() == 0) {
            shouldAddConjugate = false;
        }        
        // In the cases of an even final dimension, remove the conjugate
        // cases where we're in the middle        
        if (gridDims.back()%2 == 0 && coordsList[i].back()*2 == gridDims.back()) {
            shouldAddConjugate = false;
        }
        
        if (shouldAddConjugate) {
            if (nTermsIncluded<nTermsToInclude) {
                // Add conjugate term for most recent entry
                coeffList[i] *= (U)2.0;
                nTermsIncluded++;
            }
        }
        
        // We've added a numerical term
        nTermsNumerical++;
    }
    
    // Size down to only the number of "effective" terms needed
    coordsList.resize(nTermsNumerical);
    coeffList.resize(nTermsNumerical);
    
    // Name of dataset within this database
    SpectralDatasetID dsetID = SpectralDatasetID(dsetBasename, componentIdxUint);
    
    // Store dataset
    dsets[dsetID] = SpectralDataset<U>(coordsList, coeffList);
}

/**
 * @brief Load all of the components from a dataset
 * @param dbfile
 * @param dsetBasename
 */
template <typename U>
void SpectralDatabase<U>::loadDataset(HDF5MPIHandler& dbfile, std::string dsetBasename, unsigned int nTerms, unsigned int refineMult) {
    // Get the coefficients dataset
    std::string dsetNameCoeffs = getCoeffsName(dsetBasename);
    hid_t dsetCoeffs = dbfile.getDataset(dsetNameCoeffs);
    
    /* coeffsDataDims has contents [compDims[0],compDims[1],...,compDims[M-1],nCoeffs]
     * where M is the dimension of the base quantity that is being transformed:
     * M=0 => scalar, M=1 => vector, M=2 => 2nd order tensor, etc.
     */
    std::vector<hsize_t> coeffsDataDims = getDatasetDims(dsetCoeffs);
    
    /*  compDims contains the dimensions of the base quantity
     * empty => scalar, [N] => vector of length N, [M,N] => tensor of size MXN, etc.
     */
    std::vector<hsize_t> compDims(coeffsDataDims.begin(), coeffsDataDims.end()-1);
    
    // Number of components total
    unsigned int nComponents = 1;
    for (auto comp : compDims) {
        nComponents *= comp;
    }
    
    // Loop through all of the components
    for (unsigned int i=0; i<nComponents; i++) {
        std::vector<unsigned int> componentIdx = unflatC(i, compDims);
        this->loadDatasetSingleComponent(dbfile, dsetBasename, componentIdx, nTerms, refineMult);
    }
}

// Generate exponential table
template <typename U>
void SpectralDatabase<U>::generateExponentialTable() {
    // Exponential table max size
    unsigned int maxExpTableEntries = 0;
    for (auto dim: gridDims) {
        maxExpTableEntries += (dim-1)*(dim-1);
    }
    
    // Set up the exponential table
    std::complex<U> iUnit(0,1);
    unsigned int alignedSize = roundUpToMultiple(2*maxExpTableEntries*sizeof(U),alignment);
    expTable = std::shared_ptr<U>((U*)aligned_alloc(alignment, alignedSize), FreeDelete());    
    
    // Base entries
    for (unsigned int i=0; i<gridDims[0]; i++) {
        std::complex<U> expVal = std::exp(((U)(2*M_PI*i))*iUnit/((U)gridDims[0]));
        expTable.get()[2*i] = std::real(expVal);
        expTable.get()[2*i+1] = std::imag(expVal);
    }
    
    // Periodic extension of the table
    for (unsigned int i=gridDims[0]; i<maxExpTableEntries; i++) {
        unsigned int iOriginalDomain = i%gridDims[0];
        // Real part
        expTable.get()[i*2] = expTable.get()[iOriginalDomain*2];
        
        // Imaginary part
        expTable.get()[i*2+1] = expTable.get()[iOriginalDomain*2+1];
    }
}

// Construct from database
template <typename U>
SpectralDatabase<U>::SpectralDatabase(std::string dbFilename, std::vector<std::string> dsetBasenames, unsigned int nTerms, MPI_Comm comm, unsigned int refineMult) {    
    // Open handle to input data file
    HDF5MPIHandler dbFile(dbFilename, comm, false);
    hid_t plist = dbFile.getPropertyListTransferIndependent();
    
    // Read in number of dimensions
    hid_t dsetGridDims = dbFile.getDataset("grid_dims");
    std::vector<hsize_t> nDimsArray = getDatasetDims(dsetGridDims);
    nDims = nDimsArray[0];
    
    // Read in grid dimensions
    std::vector<unsigned short int> gridDimsBuffer(nDims);
    readSingleHDF5Array(dsetGridDims, plist, nDimsArray, gridDimsBuffer.data());
    gridDims = std::vector<unsigned int>(gridDimsBuffer.begin(), gridDimsBuffer.end());

    // Read in grid bounds
    hid_t dsetGridStarts = dbFile.getDataset("grid_starts");
    hid_t dsetGridEnds = dbFile.getDataset("grid_ends");
    gridStarts = std::vector<double>(nDims);
    gridEnds = std::vector<double>(nDims);
    readSingleHDF5Array(dsetGridStarts, plist, nDimsArray, gridStarts.data());
    readSingleHDF5Array(dsetGridEnds, plist, nDimsArray, gridEnds.data());
    
    // Calculate grid lengths and step size
    gridLengths = std::vector<double>(nDims);
    gridSteps = std::vector<double>(nDims);
    for (unsigned int i=0; i<nDims; i++) {
        gridLengths[i] = gridEnds[i] - gridStarts[i];
        gridSteps[i] = gridLengths[i]/(gridDims[i]);
    }
    
    // Refine grid dimensions
    if (refineMult == 0) {
        throw std::runtime_error("Refinement multiplier must be strictly positive.");
    }
    else {
        // Scale dimensions
        for (auto&& dim : gridDims) {
            dim *= refineMult;
        }
        
        // Scale step sizes
        for (auto&& step : gridSteps) {
            step /= refineMult;
        }
    }
    
    // Read in datasets
    for (auto dsetBasename : dsetBasenames) {
        this->loadDataset(dbFile, dsetBasename, nTerms, refineMult);
    }
    
    // Generate exponential table
    this->generateExponentialTable();
}

// Construct directly
template <typename U>
SpectralDatabase<U>::SpectralDatabase(std::vector<unsigned int> gridDims, std::vector<double> gridStarts, std::vector<double> gridEnds, std::map<SpectralDatasetID, SpectralDataset<U>> dsets) : gridDims(gridDims), gridStarts(gridStarts), gridEnds(gridEnds), dsets(dsets){
    // Derived grid properties
    nDims = gridDims.size();
    gridLengths = std::vector<double>(nDims);
    gridSteps = std::vector<double>(nDims);
    for (unsigned int i=0; i<nDims; i++) {
        gridLengths[i] = gridEnds[i] - gridStarts[i];
        gridSteps[i] = gridLengths[i]/(gridDims[i]);
    }
    
    // Generate exponential table
    this->generateExponentialTable();
}

/**
 * @brief AT END OF DAY, WAS IN PROCESS OF FILLING THIS IN, TO BE CALLED ON LINE 698
 * Should also switch IFFT->IDFT where appropriate
 */
template<typename U>
U SpectralDatabase<U>::getIDFTReal(std::string dsetBasename, std::vector<unsigned int> componentIdx, std::vector<unsigned int> spatialCoord) const {
    // Get the dataset ID
    SpectralDatasetID dsetID = SpectralDatasetID(dsetBasename, componentIdx);
    
    // Fetch the correct dataset
    std::shared_ptr<U> coeffs = dsets.at(dsetID).getCoeffs();
    std::shared_ptr<int> coords = dsets.at(dsetID).getCoords();
    unsigned int nTerms = dsets.at(dsetID).getNTerms();
    
    // Return
    if (nDims == 4) {
        return TruncatedIFFT4DSingleSquareReal(gridDims[0], expTable.get(), coords.get(), coeffs.get(), spatialCoord, nTerms);
    }
    else {
        return TruncatedIFFTNDSingleSquareReal(nDims, gridDims[0], expTable.get(), coords.get(), coeffs.get(), spatialCoord, nTerms);
    }
}

// This is for the case where the data does not have components (i.e. scalar)
template<typename U>
U SpectralDatabase<U>::getIDFTReal(std::string dsetBasename, std::vector<unsigned int> spatialCoord) const {
    // Empty component
    std::vector<unsigned int> componentIdx;
    
    // Return
    return this->getIDFTReal(dsetBasename, componentIdx, spatialCoord);
}


template<typename U>
U SpectralDatabaseUnified<U>::getIDFTReal(SpectralDatasetID dsetID, std::vector<unsigned int> spatialCoord) const {
    // Fetch the correct dataset
    std::shared_ptr<U> coeffsDset = coeffs.at(dsetID);
    
    // Perform the DFT
    if (nDims == 4) {
        return TruncatedIFFT4DSingleSquareReal(gridDims[0], coords.get(), coeffsDset.get(), spatialCoord, nTerms);
    }
    else {
        return TruncatedIFFTNDSingleSquareReal((unsigned int)nDims, gridDims[0], coords.get(), coeffsDset.get(), spatialCoord, nTerms);
    }
}
 
template<typename U>
U SpectralDatabaseUnified<U>::getIDFTReal(std::string dsetFullname, std::vector<unsigned int> spatialCoord) const {
    SpectralDatasetID dsetID = SpectralDatasetID(dsetFullname);
    return this->getIDFTReal(dsetID, spatialCoord);
}

template<typename U>
U SpectralDatabaseUnified<U>::getIDFTReal(std::string dsetBasename, std::vector<unsigned int> componentIdx, std::vector<unsigned int> spatialCoord) const {
    SpectralDatasetID dsetID = SpectralDatasetID(dsetBasename, componentIdx);
    return this->getIDFTReal(dsetID, spatialCoord);
}

///////////////////////////////
// SPECTRAL DATABASE UNIFIED //
///////////////////////////////

/**
 * @brief Load the given datasets
 * @param dbfile
 * @param dsetBasename
 * @param componentIdxUint
 */
template <typename U>
void SpectralDatabaseUnified<U>::loadDatasets(HDF5MPIHandler& dbfile, std::vector<SpectralDatasetID> dsetIDs, unsigned int nTerms, unsigned int refineMult) {
    // Number of datasets
    unsigned int nDatasets = dsetIDs.size();
    
    // Property list for reading in data
    hid_t plist_in = dbfile.getPropertyListTransferIndependent();
    
    // COORDINATES
    
    // Get coordinates dataset
    std::string dsetNameCoords = HPP_DEFAULT_UNIFIED_COORDS_NAME;
    hid_t dsetCoords = dbfile.getDataset(dsetNameCoords);
    std::vector<hsize_t> dsetCoordsDims = getDatasetDims(dsetCoords);
    unsigned int nDims = dsetCoordsDims.back();
    unsigned int nCoeffs = dsetCoordsDims.end()[-2];
    if (nTerms > nCoeffs) throw std::runtime_error("Requested more terms than exist in the dataset.");
    unsigned int nTermsToInclude = std::min(nTerms, nCoeffs);    
    
    // All reads are of scalars
    std::vector<hsize_t> componentIdx;
    
    // Read in coordinates dataset
    std::vector<unsigned short int> coordsBuffer(nTermsToInclude*nDims);  
    HDFReadWriteParamsC coordsReadParams;
    coordsReadParams.dataOffset = componentIdx;
    coordsReadParams.dataOffset.push_back(0);
    coordsReadParams.dataOffset.push_back(0);
    coordsReadParams.dataCount = std::vector<hsize_t>(componentIdx.size(), 1);
    coordsReadParams.dataCount.push_back(nTermsToInclude);
    coordsReadParams.dataCount.push_back(nDims);
    coordsReadParams.datatype = H5Dget_type(dsetCoords);
    readHDF5SimpleArray(dsetCoords, plist_in, coordsReadParams, coordsBuffer.data());    
    
    // COEFFICIENTS
    std::vector<std::vector<hdf_complex_t>> coeffsBuffers(nDatasets);    
    for (unsigned int iDset=0; iDset<nDatasets; iDset++) {
        // Prepare read buffer
        coeffsBuffers[iDset].resize(nTermsToInclude);
        
        // Determine name of coefficients dataset
        std::string dsetNameCoeffs = getDefaultCoeffsDatasetName(dsetIDs[iDset]);
        
        // Open datasets    
        hid_t dsetCoeffs = dbfile.getDataset(dsetNameCoeffs);
        
        // Get dimensions    
        std::vector<hsize_t> dsetCoeffsDims = getDatasetDims(dsetCoeffs);        
        if (nCoeffs != dsetCoeffsDims.back()) throw std::runtime_error("Mismatch in dimensions of coeffs and coords.");
        
        // Read in raw data        
        HDFReadWriteParamsC coeffsReadParams;
        coeffsReadParams.dataOffset = componentIdx;
        coeffsReadParams.dataOffset.push_back(0);
        coeffsReadParams.dataCount = std::vector<hsize_t>(componentIdx.size(), 1);
        coeffsReadParams.dataCount.push_back(nTermsToInclude);
        coeffsReadParams.datatype = H5Dget_type(dsetCoeffs);    
        readHDF5SimpleArray(dsetCoeffs, plist_in, coeffsReadParams, coeffsBuffers[iDset].data());
    }   
    
    // Convert coords and coeffs
    std::vector<std::vector<unsigned int>> coordsList(nTermsToInclude);
    std::vector<std::vector<std::complex<U>>> coeffLists(nDatasets);
    for (auto&& coeffList : coeffLists) {
        coeffList.resize(nTermsToInclude);
    }
    
    // The number of actual numerical terms is less than the raw number requested,
    // because using the fact that the data is real, we can simply double
    // the terms that are supposed to have their conjugates included
    unsigned int nTermsIncluded = 0; //number of analytic terms that have been included
    unsigned int nTermsNumerical = 0; //number of numerical terms required to capture their effect
    for (unsigned int i=0; nTermsIncluded<nTermsToInclude; i++){
        // Coords
        coordsList[i].resize(nDims);
        unsigned short int *start = coordsBuffer.data()+i*nDims;
        unsigned short int *end = start+nDims;//Note that "end" is the address one *after* the final element
        std::copy(start, end, coordsList[i].data());
        
        // Spectral refinement
        for (unsigned int j=0; j<nDims; j++) {
            // Insert zeros in the middle
            if (coordsList[i][j] >= (gridDims[j]/refineMult)/2) {
                coordsList[i][j] += gridDims[j]-(gridDims[j]/refineMult);
            }
        }
        
        // Add complex conjugates
        bool shouldAddConjugate = true;
        
        // In the case where we're double counting the final dimension
        if (coordsList[i].back() == 0) {
            shouldAddConjugate = false;
        }        
        // In the cases of an even final dimension, remove the conjugate
        // cases where we're in the middle        
        if (gridDims.back()%2 == 0 && coordsList[i].back()*2 == gridDims.back()) {
            shouldAddConjugate = false;
        }
        
        // Coeffs
        for (unsigned int iDset = 0; iDset < nDatasets; iDset++) {
            hdf_complex_t rawVal = coeffsBuffers[iDset][i];
            coeffLists[iDset][i] = std::complex<U>((U)rawVal.r, (U)rawVal.i);
        }
        nTermsIncluded++;
        
        if (shouldAddConjugate) {
            if (nTermsIncluded<nTermsToInclude) {
                // Add conjugate term for most recent entry
                for (auto&& coeffList : coeffLists) {
                    coeffList[i] *= (U)2.0;
                }
                nTermsIncluded++;
            }
        }
        
        // We've added a numerical term
        nTermsNumerical++;
    }
    
    // Set the correct number of effective terms
    this->nTerms = nTermsNumerical;
    
    // Size down to the new number of terms
    coordsList.resize(this->nTerms);
    for (auto&& coeffList : coeffLists) {
        coeffList.resize(this->nTerms);
    }
    
    // Store datasets
    nDims = coordsList[0].size();
    
    // Allocate aligned memory for coordinates
    unsigned int alignedSize = roundUpToMultiple(this->nTerms*nDims*sizeof(int), alignment);
    coords = std::shared_ptr<int>((int*)aligned_alloc(alignment, alignedSize), FreeDelete());
    
    // Populate coordinates
    for (unsigned int iTerm=0; iTerm<this->nTerms; iTerm++) {
        std::copy(coordsList[iTerm].begin(), coordsList[iTerm].end(), coords.get()+nDims*iTerm);
    }
    
    // Allocate aligned memory for coefficients and populate
    for (unsigned int iDset = 0; iDset < nDatasets; iDset++) {
        // Dataset ID
        SpectralDatasetID dsetID = dsetIDs[iDset];
        
        // Allocate aligned memory
        alignedSize = roundUpToMultiple(2*this->nTerms*sizeof(U), alignment);
        coeffs[dsetID] = std::shared_ptr<U>((U*)aligned_alloc(alignment, alignedSize), FreeDelete());
        
        for (unsigned int iTerm=0; iTerm<this->nTerms; iTerm++) {
            // Copy across terms
            coeffs[dsetID].get()[iTerm*2] = std::real(coeffLists[iDset][iTerm]);
            coeffs[dsetID].get()[iTerm*2+1] = std::imag(coeffLists[iDset][iTerm]);
        }
    }
}

// Construct from database with given dataset IDs
template <typename U>
void SpectralDatabaseUnified<U>::constructFromFile(std::string dbFilename, std::vector<SpectralDatasetID> dsetIDs, unsigned int nTerms, MPI_Comm comm, unsigned int refineMult){    
    // Open handle to input data file
    HDF5MPIHandler dbFile(dbFilename, comm, false);
    hid_t plist = dbFile.getPropertyListTransferIndependent();
    
    // Read in number of dimensions
    hid_t dsetGridDims = dbFile.getDataset("grid_dims");
    std::vector<hsize_t> nDimsArray = getDatasetDims(dsetGridDims);
    nDims = nDimsArray[0];
    
    // Read in grid dimensions
    std::vector<unsigned short int> gridDimsBuffer(nDims);
    readSingleHDF5Array(dsetGridDims, plist, nDimsArray, gridDimsBuffer.data());
    gridDims = std::vector<unsigned int>(gridDimsBuffer.begin(), gridDimsBuffer.end());

    // Read in grid bounds
    hid_t dsetGridStarts = dbFile.getDataset("grid_starts");
    hid_t dsetGridEnds = dbFile.getDataset("grid_ends");
    gridStarts = std::vector<double>(nDims);
    gridEnds = std::vector<double>(nDims);
    readSingleHDF5Array(dsetGridStarts, plist, nDimsArray, gridStarts.data());
    readSingleHDF5Array(dsetGridEnds, plist, nDimsArray, gridEnds.data());
    
    // Calculate grid lengths and step size
    gridLengths = std::vector<double>(nDims);
    gridSteps = std::vector<double>(nDims);
    for (unsigned int i=0; i<nDims; i++) {
        gridLengths[i] = gridEnds[i] - gridStarts[i];
        gridSteps[i] = gridLengths[i]/(gridDims[i]);
    }
    
    // Refine grid dimensions
    if (refineMult == 0) {
        throw std::runtime_error("Refinement multiplier must be strictly positive.");
    }
    else {
        // Scale dimensions
        for (auto&& dim : gridDims) {
            dim *= refineMult;
        }
        
        // Scale step sizes
        for (auto&& step : gridSteps) {
            step /= refineMult;
        }
    }
    
    // Read in datasets
    this->loadDatasets(dbFile, dsetIDs, nTerms, refineMult);
    
    // Generate exponential table
    //this->generateExponentialTable();
}

// Construct from database with given dataset IDs
template <typename U>
SpectralDatabaseUnified<U>::SpectralDatabaseUnified(std::string dbFilename, std::vector<SpectralDatasetID> dsetIDs, unsigned int nTerms, MPI_Comm comm, unsigned int refineMult){    
    this->constructFromFile(dbFilename, dsetIDs, nTerms, comm, refineMult);
}

// Construct from database with discovered dataset IDs
template <typename U>
SpectralDatabaseUnified<U>::SpectralDatabaseUnified(std::string dbFilename, unsigned int nTerms, MPI_Comm comm, unsigned int refineMult) {
    HDF5MPIHandler spectralFile(dbFilename, comm, false);
    std::vector<hpp::SpectralDatasetID> dsetIDs;
    for (auto dsetCoeffsName : spectralFile.getDatasetNames()) {
        size_t startOfCoeffsSuffix = dsetCoeffsName.rfind(HPP_DEFAULT_COEFFS_SUFFIX);
        if (startOfCoeffsSuffix != std::string::npos) {
            std::string dsetName = dsetCoeffsName.substr(0,startOfCoeffsSuffix);
            dsetIDs.push_back(hpp::SpectralDatasetID(dsetName));
        }
    }
    this->constructFromFile(dbFilename, dsetIDs, nTerms, comm, refineMult);
}

// Specific instantiations of spectral databases
template class SpectralDataset<float>;
template class SpectralDataset<double>;
template class SpectralDatabase<float>;
template class SpectralDatabase<double>;
template class SpectralDatabaseUnified<float>;
template class SpectralDatabaseUnified<double>;

void evaluateSpectralCompressionErrorFull(std::string rawDBName, std::string spectralDBName, std::string errorDBName, unsigned int nTermsMax, std::string outFilename, MPI_Comm comm) {
    // Open files for reading
    HDF5MPIHandler rawFile(rawDBName, comm, false);
    hid_t plistRaw = rawFile.getPropertyListTransferIndependent();
    HDF5MPIHandler spectralFile(spectralDBName, comm, false);
    
    // Spatial grid dimensions
    hid_t dsetGridDims = rawFile.getDataset("grid_dims");
    std::vector<hsize_t> nDimsArray = getDatasetDims(dsetGridDims);
    hsize_t nDims = nDimsArray[0];
    std::vector<unsigned short int> spatialDimsBuffer(nDims);
    readSingleHDF5Array(dsetGridDims, plistRaw, nDimsArray, spatialDimsBuffer.data());
    std::vector<unsigned int> spatialDims(spatialDimsBuffer.begin(), spatialDimsBuffer.end());
    unsigned int nSpatialPoints = 1;
    for (unsigned int i=0; i<nDims; i++) {
        nSpatialPoints *= spatialDims[i];
    }
    
    // For analysing the full dataset, break up spatial points amongst the processes
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    unsigned int nSpatialPointsLocal = nSpatialPoints/comm_size;
    if (nSpatialPoints % comm_size != 0) nSpatialPointsLocal++;
    unsigned int spatialStart = comm_rank*nSpatialPointsLocal;
    unsigned int spatialEnd = spatialStart+nSpatialPointsLocal-1;
    
    // Spatial grid lengths
    hid_t dsetGridStarts = rawFile.getDataset("grid_starts");
    hid_t dsetGridEnds = rawFile.getDataset("grid_ends");
    std::vector<double> gridStarts(nDims);
    std::vector<double> gridEnds(nDims);
    std::vector<double> spatialLengths(nDims);
    readSingleHDF5Array(dsetGridStarts, plistRaw, nDimsArray, gridStarts.data());
    readSingleHDF5Array(dsetGridEnds, plistRaw, nDimsArray, gridEnds.data());
    for (unsigned int i=0; i<nDims; i++) {
        spatialLengths[i] = gridEnds[i] - gridStarts[i];
    }
    
    // Exponential tables
    std::vector<std::vector<std::complex<double>>> expTables(nDims);
    std::complex<double> iUnit(0, 1);
    for (unsigned int dim=0; dim<nDims; dim++) {
        expTables[dim].resize(spatialDims[dim]);
        for (unsigned int i=0; i<spatialDims[dim]; i++) {
            expTables[dim][i] = std::exp(((double)(2*M_PI*i))*iUnit/((double)spatialDims[dim]));
        }
    }
    
    // Get dataset names and sizes
    std::vector<std::string> inputDatasetNames = rawFile.getDatasetNames();
    std::vector<std::string> rawDatasetNames;
    std::vector<std::string> coordsDatasetNames;
    std::vector<std::string> coeffsDatasetNames;
    std::vector<std::vector<hsize_t>> primaryQuantityDims;
    for (auto&& name : inputDatasetNames) {
        // Avoid metadata
        if (name.substr(0,5) != "grid_") {
            rawDatasetNames.push_back(name);
            coordsDatasetNames.push_back(name+"_coords");
            coeffsDatasetNames.push_back(name+"_vals");
            hid_t rawDataset = rawFile.getDataset(name);
            std::vector<hsize_t> rawDatasetDims = getDatasetDims(rawDataset);
            std::vector<hsize_t> primaryQuantityDim;
            for (unsigned int i=0; i<rawDatasetDims.size() - nDims; i++) {
                primaryQuantityDim.push_back(rawDatasetDims[i]);
            }
            primaryQuantityDims.push_back(primaryQuantityDim);
        }
    }
    unsigned int nRawDatasets = rawDatasetNames.size();
    
    // Cap number of terms to maximum available
    hid_t someCoeffsDataset = spectralFile.getDataset(coeffsDatasetNames[0]);
    std::vector<hsize_t> someCoeffsDatasetDims = getDatasetDims(someCoeffsDataset);
    unsigned int nTermsAvailable = someCoeffsDatasetDims.back();
    nTermsMax = std::min(nTermsMax, nTermsAvailable);
    
    // Spatial read indices
    // Extends read range if necessary to make clean breaks along first dimension
    std::vector<unsigned int> startIdx = unflatC(spatialStart, spatialDims);
    std::vector<unsigned int> endIdx = unflatC(spatialEnd, spatialDims);
    std::vector<unsigned int> bufferStartIdx = startIdx;
    std::vector<unsigned int> bufferEndIdx = endIdx;
    for (unsigned int i=1; i<bufferStartIdx.size(); i++) {
        bufferStartIdx[i] = 0;
        bufferEndIdx[i] = spatialDims[i]-1;
    }
    unsigned int bufferStartFlatIdx = flatC(bufferStartIdx, spatialDims);
    
    std::vector<unsigned int> bufferCount(spatialDims.begin(), spatialDims.end());
    bufferCount[0] = bufferEndIdx[0] - bufferStartIdx[0] + 1;    
    
    unsigned int rawBufferSize = 1;
    for (auto count : bufferCount) {
        rawBufferSize *= count;
    }
    std::vector<double> rawValsBuffer(rawBufferSize);
    double *rawVals = rawValsBuffer.data() + (spatialStart - bufferStartFlatIdx);
    
    HDFReadWriteParamsC rawReadParamsBase;
    rawReadParamsBase.dataOffset = std::vector<hsize_t>(bufferStartIdx.begin(), bufferStartIdx.end());
    rawReadParamsBase.dataCount = std::vector<hsize_t>(bufferCount.begin(), bufferCount.end());
    rawReadParamsBase.datatype = getHDF5TypeC<double>();
    
    // Process the datasets
    for (unsigned int iDset=0; iDset<nRawDatasets; iDset++) {
        std::vector<hsize_t> primaryQuantityDim = primaryQuantityDims[iDset];
        unsigned int primaryQuantityRank = primaryQuantityDim.size();
        std::vector<std::vector<unsigned int>> spectralCoordsList;
        std::vector<std::complex<double>> coeffs;
        std::string dsetNameCoords = coordsDatasetNames[iDset];
        std::string dsetNameCoeffs = coeffsDatasetNames[iDset];
        std::string dsetNameRaw = rawDatasetNames[iDset];
        hsize_t nComponents = std::accumulate(primaryQuantityDim.begin(), primaryQuantityDim.end(), 1, std::multiplies<hsize_t>());
        // Loop through components
        for (hsize_t iComp=0; iComp<nComponents; iComp++) {            
            // Get the index of this component
            std::vector<hsize_t> componentIdx;
            if (primaryQuantityRank != 0) {
                componentIdx = unflatC(iComp, primaryQuantityDim);
            }
            
            // Read parameters
            HDFReadWriteParamsC rawReadParams;
            rawReadParams.dataOffset = componentIdx;
            rawReadParams.dataOffset.insert(rawReadParams.dataOffset.end(), rawReadParamsBase.dataOffset.begin(), rawReadParamsBase.dataOffset.end());
            rawReadParams.dataCount = std::vector<hsize_t>(primaryQuantityRank, 1);
            rawReadParams.dataCount.insert(rawReadParams.dataCount.end(), rawReadParamsBase.dataCount.begin(), rawReadParamsBase.dataCount.end());
            rawReadParams.datatype = rawReadParamsBase.datatype;
            
            // Load the raw database entries
            hid_t dsetRaw = rawFile.getDataset(dsetNameRaw);            
            readHDF5SimpleArray(dsetRaw, plistRaw, rawReadParams, rawValsBuffer.data());
            
            // Load the spectral database for the component
            loadSpectralDatabase(spectralFile, dsetNameCoords, dsetNameCoeffs, componentIdx, spectralCoordsList, coeffs, nTermsMax);
            unsigned int nTermsNumerical = coeffs.size();
            
            // Loop through spatial points to get the error at each point
            double minError = std::numeric_limits<double>::max();
            double maxError = std::numeric_limits<double>::min();
            double errorSum = 0.0;
            double errorSquareSum = 0.0;
            for (unsigned int iLocal=0; iLocal<nSpatialPointsLocal; iLocal++) {
                // Get the coordinate in the spatial grid
                unsigned int flatIdx = spatialStart + iLocal;
                std::vector<unsigned int> spatialCoord = unflatC(flatIdx, spatialDims);
                
                // Get the spectral value
                std::complex<double> spectralVal = TruncatedIFFTNDSingleSquare(spatialDims[0], expTables[0], spectralCoordsList, coeffs, spatialCoord, nTermsNumerical);
                
                // Compare with raw value
                double realSpectralVal = std::real(spectralVal);
                double error = std::abs(realSpectralVal - rawVals[iLocal]);
                minError = std::min(error, minError);
                maxError = std::max(error, maxError);
                errorSum += error;
                errorSquareSum += std::pow(error, 2.0);
            }

            // Gather the error statistics on the root
            std::vector<double> minErrors = MPIConcatOnRoot(minError, comm);
            std::vector<double> maxErrors = MPIConcatOnRoot(maxError, comm);
            std::vector<double> errorSums = MPIConcatOnRoot(errorSum, comm);
            std::vector<double> errorSquareSums = MPIConcatOnRoot(errorSquareSum, comm);
            
            // Report errors on root
            if (comm_rank == 0) {
                // Gather values
                minError = *(std::min_element(minErrors.begin(), minErrors.end()));
                maxError = *(std::max_element(maxErrors.begin(), maxErrors.end()));
                errorSum = std::accumulate(errorSums.begin(), errorSums.end(), 0.0);
                errorSquareSum = std::accumulate(errorSquareSums.begin(), errorSquareSums.end(), 0.0);
                
                // Norms
                double l1Error = errorSum/nSpatialPoints;
                double l2Error = std::sqrt(errorSquareSum/nSpatialPoints);
                double lInfError = maxError;
                
                // Errors
                std::cout << dsetNameRaw;
                operator<<(std::cout, componentIdx);
                std::cout << ":" << std::endl;                
                std::cout << "L1 error = " << l1Error << std::endl;
                std::cout << "L2 error = " << l2Error << std::endl;
                std::cout << "LInf error = " << lInfError << std::endl;
            }
        }
    }    
}

void evaluateSpectralCompressionErrorFullUnified(std::string rawDBName, std::string spectralDBName, std::string errorDBName, unsigned int nTermsMax, std::string outFilename, MPI_Comm comm) {
    throw std::runtime_error("Not implemented yet. Currently a copy of non-unified version.");  
    
    // Open files for reading
    HDF5MPIHandler rawFile(rawDBName, comm, false);
    hid_t plistRaw = rawFile.getPropertyListTransferIndependent();
    HDF5MPIHandler spectralFile(spectralDBName, comm, false);
    
    // Spatial grid dimensions
    hid_t dsetGridDims = rawFile.getDataset("grid_dims");
    std::vector<hsize_t> nDimsArray = getDatasetDims(dsetGridDims);
    hsize_t nDims = nDimsArray[0];
    std::vector<unsigned short int> spatialDimsBuffer(nDims);
    readSingleHDF5Array(dsetGridDims, plistRaw, nDimsArray, spatialDimsBuffer.data());
    std::vector<unsigned int> spatialDims(spatialDimsBuffer.begin(), spatialDimsBuffer.end());
    unsigned int nSpatialPoints = 1;
    for (unsigned int i=0; i<nDims; i++) {
        nSpatialPoints *= spatialDims[i];
    }
    
    // For analysing the full dataset, break up spatial points amongst the processes
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    unsigned int nSpatialPointsLocal = nSpatialPoints/comm_size;
    if (nSpatialPoints % comm_size != 0) nSpatialPointsLocal++;
    unsigned int spatialStart = comm_rank*nSpatialPointsLocal;
    unsigned int spatialEnd = spatialStart+nSpatialPointsLocal-1;
    
    // Spatial grid lengths
    hid_t dsetGridStarts = rawFile.getDataset("grid_starts");
    hid_t dsetGridEnds = rawFile.getDataset("grid_ends");
    std::vector<double> gridStarts(nDims);
    std::vector<double> gridEnds(nDims);
    std::vector<double> spatialLengths(nDims);
    readSingleHDF5Array(dsetGridStarts, plistRaw, nDimsArray, gridStarts.data());
    readSingleHDF5Array(dsetGridEnds, plistRaw, nDimsArray, gridEnds.data());
    for (unsigned int i=0; i<nDims; i++) {
        spatialLengths[i] = gridEnds[i] - gridStarts[i];
    }
    
    // Exponential tables
    std::vector<std::vector<std::complex<double>>> expTables(nDims);
    std::complex<double> iUnit(0, 1);
    for (unsigned int dim=0; dim<nDims; dim++) {
        expTables[dim].resize(spatialDims[dim]);
        for (unsigned int i=0; i<spatialDims[dim]; i++) {
            expTables[dim][i] = std::exp(((double)(2*M_PI*i))*iUnit/((double)spatialDims[dim]));
        }
    }
    
    // Get dataset names and sizes
    std::vector<std::string> inputDatasetNames = rawFile.getDatasetNames();
    std::vector<std::string> rawDatasetNames;
    std::vector<std::string> coordsDatasetNames;
    std::vector<std::string> coeffsDatasetNames;
    std::vector<std::vector<hsize_t>> primaryQuantityDims;
    for (auto&& name : inputDatasetNames) {
        // Avoid metadata
        if (name.substr(0,5) != "grid_") {
            rawDatasetNames.push_back(name);
            coordsDatasetNames.push_back(name+"_coords");
            coeffsDatasetNames.push_back(name+"_vals");
            hid_t rawDataset = rawFile.getDataset(name);
            std::vector<hsize_t> rawDatasetDims = getDatasetDims(rawDataset);
            std::vector<hsize_t> primaryQuantityDim;
            for (unsigned int i=0; i<rawDatasetDims.size() - nDims; i++) {
                primaryQuantityDim.push_back(rawDatasetDims[i]);
            }
            primaryQuantityDims.push_back(primaryQuantityDim);
        }
    }
    unsigned int nRawDatasets = rawDatasetNames.size();
    
    // Cap number of terms to maximum available
    hid_t someCoeffsDataset = spectralFile.getDataset(coeffsDatasetNames[0]);
    std::vector<hsize_t> someCoeffsDatasetDims = getDatasetDims(someCoeffsDataset);
    unsigned int nTermsAvailable = someCoeffsDatasetDims.back();
    nTermsMax = std::min(nTermsMax, nTermsAvailable);
    
    // Spatial read indices
    // Extends read range if necessary to make clean breaks along first dimension
    std::vector<unsigned int> startIdx = unflatC(spatialStart, spatialDims);
    std::vector<unsigned int> endIdx = unflatC(spatialEnd, spatialDims);
    std::vector<unsigned int> bufferStartIdx = startIdx;
    std::vector<unsigned int> bufferEndIdx = endIdx;
    for (unsigned int i=1; i<bufferStartIdx.size(); i++) {
        bufferStartIdx[i] = 0;
        bufferEndIdx[i] = spatialDims[i]-1;
    }
    unsigned int bufferStartFlatIdx = flatC(bufferStartIdx, spatialDims);
    
    std::vector<unsigned int> bufferCount(spatialDims.begin(), spatialDims.end());
    bufferCount[0] = bufferEndIdx[0] - bufferStartIdx[0] + 1;    
    
    unsigned int rawBufferSize = 1;
    for (auto count : bufferCount) {
        rawBufferSize *= count;
    }
    std::vector<double> rawValsBuffer(rawBufferSize);
    double *rawVals = rawValsBuffer.data() + (spatialStart - bufferStartFlatIdx);
    
    HDFReadWriteParamsC rawReadParamsBase;
    rawReadParamsBase.dataOffset = std::vector<hsize_t>(bufferStartIdx.begin(), bufferStartIdx.end());
    rawReadParamsBase.dataCount = std::vector<hsize_t>(bufferCount.begin(), bufferCount.end());
    rawReadParamsBase.datatype = getHDF5TypeC<double>();
    
    // Process the datasets
    for (unsigned int iDset=0; iDset<nRawDatasets; iDset++) {
        std::vector<hsize_t> primaryQuantityDim = primaryQuantityDims[iDset];
        unsigned int primaryQuantityRank = primaryQuantityDim.size();
        std::vector<std::vector<unsigned int>> spectralCoordsList;
        std::vector<std::complex<double>> coeffs;
        std::string dsetNameCoords = coordsDatasetNames[iDset];
        std::string dsetNameCoeffs = coeffsDatasetNames[iDset];
        std::string dsetNameRaw = rawDatasetNames[iDset];
        hsize_t nComponents = std::accumulate(primaryQuantityDim.begin(), primaryQuantityDim.end(), 1, std::multiplies<hsize_t>());
        // Loop through components
        for (hsize_t iComp=0; iComp<nComponents; iComp++) {            
            // Get the index of this component
            std::vector<hsize_t> componentIdx;
            if (primaryQuantityRank != 0) {
                componentIdx = unflatC(iComp, primaryQuantityDim);
            }
            
            // Read parameters
            HDFReadWriteParamsC rawReadParams;
            rawReadParams.dataOffset = componentIdx;
            rawReadParams.dataOffset.insert(rawReadParams.dataOffset.end(), rawReadParamsBase.dataOffset.begin(), rawReadParamsBase.dataOffset.end());
            rawReadParams.dataCount = std::vector<hsize_t>(primaryQuantityRank, 1);
            rawReadParams.dataCount.insert(rawReadParams.dataCount.end(), rawReadParamsBase.dataCount.begin(), rawReadParamsBase.dataCount.end());
            rawReadParams.datatype = rawReadParamsBase.datatype;
            
            // Load the raw database entries
            hid_t dsetRaw = rawFile.getDataset(dsetNameRaw);            
            readHDF5SimpleArray(dsetRaw, plistRaw, rawReadParams, rawValsBuffer.data());
            
            // Load the spectral database for the component
            loadSpectralDatabase(spectralFile, dsetNameCoords, dsetNameCoeffs, componentIdx, spectralCoordsList, coeffs, nTermsMax);
            unsigned int nTermsNumerical = coeffs.size();
            
            // Loop through spatial points to get the error at each point
            double minError = std::numeric_limits<double>::max();
            double maxError = std::numeric_limits<double>::min();
            double errorSum = 0.0;
            double errorSquareSum = 0.0;
            for (unsigned int iLocal=0; iLocal<nSpatialPointsLocal; iLocal++) {
                // Get the coordinate in the spatial grid
                unsigned int flatIdx = spatialStart + iLocal;
                std::vector<unsigned int> spatialCoord = unflatC(flatIdx, spatialDims);
                
                // Get the spectral value
                std::complex<double> spectralVal = TruncatedIFFTNDSingleSquare(spatialDims[0], expTables[0], spectralCoordsList, coeffs, spatialCoord, nTermsNumerical);
                
                // Compare with raw value
                double realSpectralVal = std::real(spectralVal);
                double error = std::abs(realSpectralVal - rawVals[iLocal]);
                minError = std::min(error, minError);
                maxError = std::max(error, maxError);
                errorSum += error;
                errorSquareSum += std::pow(error, 2.0);
            }

            // Gather the error statistics on the root
            std::vector<double> minErrors = MPIConcatOnRoot(minError, comm);
            std::vector<double> maxErrors = MPIConcatOnRoot(maxError, comm);
            std::vector<double> errorSums = MPIConcatOnRoot(errorSum, comm);
            std::vector<double> errorSquareSums = MPIConcatOnRoot(errorSquareSum, comm);
            
            // Report errors on root
            if (comm_rank == 0) {
                // Gather values
                minError = *(std::min_element(minErrors.begin(), minErrors.end()));
                maxError = *(std::max_element(maxErrors.begin(), maxErrors.end()));
                errorSum = std::accumulate(errorSums.begin(), errorSums.end(), 0.0);
                errorSquareSum = std::accumulate(errorSquareSums.begin(), errorSquareSums.end(), 0.0);
                
                // Norms
                double l1Error = errorSum/nSpatialPoints;
                double l2Error = std::sqrt(errorSquareSum/nSpatialPoints);
                double lInfError = maxError;
                
                // Errors
                std::cout << dsetNameRaw;
                operator<<(std::cout, componentIdx);
                std::cout << ":" << std::endl;                
                std::cout << "L1 error = " << l1Error << std::endl;
                std::cout << "L2 error = " << l2Error << std::endl;
                std::cout << "LInf error = " << lInfError << std::endl;
            }
        }
    }    
}

/**
 * @brief Evaluate the error incurred in the spectral reresentation of a database
 * @param rawDBName
 * @param spectralDBName
 * @param errorDBName
 * @param nTerms
 * @param axisSlice a vector of length N, where N is the dimension of the dataset, containing [axis, otherCoord0, otherCoord1, ..., otherCoordN-1], where axis specifies the axis along which a slice is to be taken, and the values otherCoord0, ..., otherCoordN-1 specify in order what the fixed values of the coordinates should be in the other dimensions. For example, an axis slice of [2, 19, 3, 5] specifies that the slice is to be taken along axis/dimension "2", that is, the third axis. The following numbers then dictate that the fixed coordinates for the other axes should be:
    - 19 for axis 0
    - 3 for axis 1
    - 5 for axis 3
 * @param comm
 */
void evaluateSpectralCompressionErrorAxisSlice(std::string rawDBName, std::string spectralDBName, std::vector<std::string> dsetBasenames, unsigned int nTermsMax, unsigned int refineMult, std::vector<int> axisSlice, std::string outFilename, MPI_Comm comm) {
    // Only process this function on root in this implementation
    int comm_size, comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    if (comm_rank == 0) {
        comm = MPI_COMM_SELF;
    }
    else {
        return;
    }
    
    // This is simply size 1 and rank 0 now
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Spectral database
    SpectralDatabase<double> dbSpectral(spectralDBName, dsetBasenames, nTermsMax, comm, refineMult);
    
    // Open raw database file for reading
    HDF5MPIHandler rawFile(rawDBName, comm, false);
    hid_t plistRaw = rawFile.getPropertyListTransferIndependent();
    
    // Spatial grid dimensions
    unsigned int nDims = dbSpectral.getNDims();
    std::vector<unsigned int> spatialDims = dbSpectral.getGridDims();
    unsigned int nSpatialPoints = 1;
    for (unsigned int i=0; i<nDims; i++) {
        nSpatialPoints *= spatialDims[i];
    }
    
    // Slice information
    int axis = axisSlice[0];
    std::vector<int> fixedCoords(axisSlice.begin()+1, axisSlice.end());
    
    // For analysing the full dataset, break up spatial points amongst the processes
    unsigned int nSpatialPointsLocal = nSpatialPoints/comm_size;
    if (nSpatialPoints % comm_size != 0) nSpatialPointsLocal++;
    unsigned int spatialStart = comm_rank*nSpatialPointsLocal;
    unsigned int spatialEnd = spatialStart+nSpatialPointsLocal-1;
    
    // Spatial grid lengths
    std::vector<double> gridStarts = dbSpectral.getGridStarts();
    std::vector<double> gridEnds = dbSpectral.getGridEnds();
    std::vector<double> spatialLengths = dbSpectral.getGridLengths();
    
    // Get dataset names and sizes
    std::vector<std::string> rawDatasetNames = dsetBasenames;
    unsigned int nRawDatasets = rawDatasetNames.size();
    std::vector<std::vector<hsize_t>> primaryQuantityDims;
    for (auto&& name : rawDatasetNames) {
        hid_t rawDataset = rawFile.getDataset(name);
        std::vector<hsize_t> rawDatasetDims = getDatasetDims(rawDataset);
        std::vector<hsize_t> primaryQuantityDim;
        for (unsigned int i=0; i<rawDatasetDims.size() - nDims; i++) {
            primaryQuantityDim.push_back(rawDatasetDims[i]);
        }
        primaryQuantityDims.push_back(primaryQuantityDim);
    }
    
    // Spatial read indices
    // Extends read range if necessary to make clean breaks along first dimension
    std::vector<unsigned int> startIdx = unflatC(spatialStart, spatialDims);
    std::vector<unsigned int> endIdx = unflatC(spatialEnd, spatialDims);
    std::vector<unsigned int> bufferStartIdx = startIdx;
    std::vector<unsigned int> bufferEndIdx = endIdx;
    for (unsigned int i=1; i<bufferStartIdx.size(); i++) {
        bufferStartIdx[i] = 0;
        bufferEndIdx[i] = spatialDims[i]-1;
    }
    unsigned int bufferStartFlatIdx = flatC(bufferStartIdx, spatialDims);
    
    std::vector<unsigned int> bufferCount(spatialDims.begin(), spatialDims.end());
    bufferCount[0] = bufferEndIdx[0] - bufferStartIdx[0] + 1;    
    
    unsigned int rawBufferSize = 1;
    for (auto count : bufferCount) {
        rawBufferSize *= count;
    }
    std::vector<double> rawValsBuffer(rawBufferSize);
    double *rawVals = rawValsBuffer.data() + (spatialStart - bufferStartFlatIdx);
    
    HDFReadWriteParamsC rawReadParamsBase;
    rawReadParamsBase.dataOffset = std::vector<hsize_t>(bufferStartIdx.begin(), bufferStartIdx.end());
    rawReadParamsBase.dataCount = std::vector<hsize_t>(bufferCount.begin(), bufferCount.end());
    rawReadParamsBase.datatype = getHDF5TypeC<double>();
    
    // Open file for writing results
    std::ofstream outfile(outFilename.c_str());
    
    // Process the datasets
    for (unsigned int iDset=0; iDset<nRawDatasets; iDset++) {
        std::vector<hsize_t> primaryQuantityDim = primaryQuantityDims[iDset];
        unsigned int primaryQuantityRank = primaryQuantityDim.size();
        std::vector<std::vector<unsigned int>> spectralCoordsList;
        std::vector<std::complex<double>> coeffs;
        std::string dsetBasename = dsetBasenames[iDset];
        std::string dsetNameRaw = rawDatasetNames[iDset];
        hsize_t nComponents = std::accumulate(primaryQuantityDim.begin(), primaryQuantityDim.end(), 1, std::multiplies<hsize_t>());
        // Loop through components
        for (hsize_t iComp=0; iComp<nComponents; iComp++) {            
            // Get the index of this component
            std::vector<hsize_t> componentIdx;
            if (primaryQuantityRank != 0) {
                componentIdx = unflatC(iComp, primaryQuantityDim);
            }
            
            // Read parameters
            HDFReadWriteParamsC rawReadParams;
            rawReadParams.dataOffset = componentIdx;
            rawReadParams.dataOffset.insert(rawReadParams.dataOffset.end(), rawReadParamsBase.dataOffset.begin(), rawReadParamsBase.dataOffset.end());
            rawReadParams.dataCount = std::vector<hsize_t>(primaryQuantityRank, 1);
            rawReadParams.dataCount.insert(rawReadParams.dataCount.end(), rawReadParamsBase.dataCount.begin(), rawReadParamsBase.dataCount.end());
            rawReadParams.datatype = rawReadParamsBase.datatype;
            
            // Load the raw database entries
            hid_t dsetRaw = rawFile.getDataset(dsetNameRaw);            
            readHDF5SimpleArray(dsetRaw, plistRaw, rawReadParams, rawValsBuffer.data());
            
            // Loop through the axial slice
            std::vector<double> rawValsCoord(spatialDims[axis]);
            std::vector<double> spectralValsCoord(spatialDims[axis]);            
            for (unsigned int iCoord=0; iCoord<spatialDims[axis]; iCoord++) {
                // Get the full coordinate in the spatial grid
                std::vector<unsigned int> spatialCoord(fixedCoords.begin(),fixedCoords.end());
                spatialCoord.insert(spatialCoord.begin()+axis, iCoord);
                
                // Get the spectral value
                std::vector<unsigned int> componentIdxUint(componentIdx.begin(), componentIdx.end());
                double spectralVal = dbSpectral.getIDFTReal(dsetBasename, componentIdxUint, spatialCoord);
                
                // Store values
                spectralValsCoord[iCoord] = spectralVal;
                rawValsCoord[iCoord] = rawVals[flatC(spatialCoord, spatialDims)];
            }
            
            // Write to file
            std::string dsetComponentName = dsetNameRaw+"_";
            for (auto component : componentIdx) {
                dsetComponentName += std::to_string(component);
            }
            outfile << dsetComponentName+"_raw=" << rawValsCoord << std::endl;
            outfile << dsetComponentName+"_spectral=" << spectralValsCoord << std::endl;
        }
    }
}

/**
 * @brief Evaluate the error incurred in the spectral reresentation of a database
 * @detail For the "unified" database format, with a single set of ordered coordinates
 * corresponding to all of the components.
 * @param rawDBName
 * @param spectralDBName
 * @param errorDBName
 * @param nTerms
 * @param axisSlice a vector of length N, where N is the dimension of the dataset, containing [axis, otherCoord0, otherCoord1, ..., otherCoordN-1], where axis specifies the axis along which a slice is to be taken, and the values otherCoord0, ..., otherCoordN-1 specify in order what the fixed values of the coordinates should be in the other dimensions. For example, an axis slice of [2, 19, 3, 5] specifies that the slice is to be taken along axis/dimension "2", that is, the third axis. The following numbers then dictate that the fixed coordinates for the other axes should be:
    - 19 for axis 0
    - 3 for axis 1
    - 5 for axis 3
 * @param comm
 */
void evaluateSpectralCompressionErrorAxisSliceUnified(std::string rawDBName, std::string spectralDBName, std::vector<std::string> dsetBasenames, unsigned int nTermsMax, unsigned int refineMult, std::vector<int> axisSlice, std::string outFilename, MPI_Comm comm) {
    // Only process this function on root in this implementation
    int comm_size, comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    if (comm_rank == 0) {
        comm = MPI_COMM_SELF;
    }
    else {
        return;
    }
    
    // This is simply size 1 and rank 0 now
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Open raw database file for reading
    HDF5MPIHandler rawFile(rawDBName, comm, false);
    hid_t plistRaw = rawFile.getPropertyListTransferIndependent();
    
    // Load spectral database
    hpp::SpectralDatabaseUnified<double> dbSpectral(spectralDBName, nTermsMax, comm, refineMult);
    
    // Spatial grid dimensions
    unsigned int nDims = dbSpectral.getNDims();
    std::vector<unsigned int> spatialDims = dbSpectral.getGridDims();
    unsigned int nSpatialPoints = 1;
    for (unsigned int i=0; i<nDims; i++) {
        nSpatialPoints *= spatialDims[i];
    }
    
    // Slice information
    int axis = axisSlice[0];
    std::vector<int> fixedCoords(axisSlice.begin()+1, axisSlice.end());
    
    // For analysing the full dataset, break up spatial points amongst the processes
    unsigned int nSpatialPointsLocal = nSpatialPoints/comm_size;
    if (nSpatialPoints % comm_size != 0) nSpatialPointsLocal++;
    unsigned int spatialStart = comm_rank*nSpatialPointsLocal;
    unsigned int spatialEnd = spatialStart+nSpatialPointsLocal-1;
    
    // Spatial grid lengths
    std::vector<double> gridStarts = dbSpectral.getGridStarts();
    std::vector<double> gridEnds = dbSpectral.getGridEnds();
    std::vector<double> spatialLengths = dbSpectral.getGridLengths();
    
    // Get dataset names and sizes
    std::vector<std::string> rawDatasetNames = dsetBasenames;
    unsigned int nRawDatasets = rawDatasetNames.size();
    std::vector<std::vector<hsize_t>> primaryQuantityDims;
    for (auto&& name : rawDatasetNames) {
        hid_t rawDataset = rawFile.getDataset(name);
        std::vector<hsize_t> rawDatasetDims = getDatasetDims(rawDataset);
        std::vector<hsize_t> primaryQuantityDim;
        for (unsigned int i=0; i<rawDatasetDims.size() - nDims; i++) {
            primaryQuantityDim.push_back(rawDatasetDims[i]);
        }
        primaryQuantityDims.push_back(primaryQuantityDim);
    }
    
    // Spatial read indices
    // Extends read range if necessary to make clean breaks along first dimension
    std::vector<unsigned int> startIdx = unflatC(spatialStart, spatialDims);
    std::vector<unsigned int> endIdx = unflatC(spatialEnd, spatialDims);
    std::vector<unsigned int> bufferStartIdx = startIdx;
    std::vector<unsigned int> bufferEndIdx = endIdx;
    for (unsigned int i=1; i<bufferStartIdx.size(); i++) {
        bufferStartIdx[i] = 0;
        bufferEndIdx[i] = spatialDims[i]-1;
    }
    unsigned int bufferStartFlatIdx = flatC(bufferStartIdx, spatialDims);
    
    std::vector<unsigned int> bufferCount(spatialDims.begin(), spatialDims.end());
    bufferCount[0] = bufferEndIdx[0] - bufferStartIdx[0] + 1;    
    
    unsigned int rawBufferSize = 1;
    for (auto count : bufferCount) {
        rawBufferSize *= count;
    }
    std::vector<double> rawValsBuffer(rawBufferSize);
    double *rawVals = rawValsBuffer.data() + (spatialStart - bufferStartFlatIdx);
    
    HDFReadWriteParamsC rawReadParamsBase;
    rawReadParamsBase.dataOffset = std::vector<hsize_t>(bufferStartIdx.begin(), bufferStartIdx.end());
    rawReadParamsBase.dataCount = std::vector<hsize_t>(bufferCount.begin(), bufferCount.end());
    rawReadParamsBase.datatype = getHDF5TypeC<double>();
    
    // Open file for writing results
    std::ofstream outfile(outFilename.c_str());
    
    // Process the datasets
    for (unsigned int iDset=0; iDset<nRawDatasets; iDset++) {
        std::vector<hsize_t> primaryQuantityDim = primaryQuantityDims[iDset];
        unsigned int primaryQuantityRank = primaryQuantityDim.size();
        std::vector<std::vector<unsigned int>> spectralCoordsList;
        std::vector<std::complex<double>> coeffs;
        std::string dsetBasename = dsetBasenames[iDset];
        std::string dsetNameRaw = rawDatasetNames[iDset];
        hsize_t nComponents = std::accumulate(primaryQuantityDim.begin(), primaryQuantityDim.end(), 1, std::multiplies<hsize_t>());
        // Loop through components
        for (hsize_t iComp=0; iComp<nComponents; iComp++) {            
            // Get the index of this component
            std::vector<hsize_t> componentIdx;
            if (primaryQuantityRank != 0) {
                componentIdx = unflatC(iComp, primaryQuantityDim);
            }
            
            // Continue if component is in spectral database
            // It may have been discarded due to symmetry in the represented
            // tensors or whatever else is being stored
            std::string dsetSpectralFullName = dsetBasename+getComponentSuffix(componentIdx);
            auto dsetSpectralID = hpp::SpectralDatasetID(dsetSpectralFullName);
            if (dbSpectral.hasDset(dsetSpectralID)) {
                // Read parameters
                HDFReadWriteParamsC rawReadParams;
                rawReadParams.dataOffset = componentIdx;
                rawReadParams.dataOffset.insert(rawReadParams.dataOffset.end(), rawReadParamsBase.dataOffset.begin(), rawReadParamsBase.dataOffset.end());
                rawReadParams.dataCount = std::vector<hsize_t>(primaryQuantityRank, 1);
                rawReadParams.dataCount.insert(rawReadParams.dataCount.end(), rawReadParamsBase.dataCount.begin(), rawReadParamsBase.dataCount.end());
                rawReadParams.datatype = rawReadParamsBase.datatype;
                
                // Load the raw database entries
                hid_t dsetRaw = rawFile.getDataset(dsetNameRaw);            
                readHDF5SimpleArray(dsetRaw, plistRaw, rawReadParams, rawValsBuffer.data());
                
                // Loop through the axial slice
                std::vector<double> rawValsCoord(spatialDims[axis]);
                std::vector<double> spectralValsCoord(spatialDims[axis]);            
                for (unsigned int iCoord=0; iCoord<spatialDims[axis]; iCoord++) {
                    // Get the full coordinate in the spatial grid
                    std::vector<unsigned int> spatialCoord(fixedCoords.begin(),fixedCoords.end());
                    spatialCoord.insert(spatialCoord.begin()+axis, iCoord);
                    
                    // Get the spectral value
                    std::vector<unsigned int> componentIdxUint(componentIdx.begin(), componentIdx.end());
                    double spectralVal = dbSpectral.getIDFTReal(dsetSpectralFullName, spatialCoord);
                    
                    // Store values
                    spectralValsCoord[iCoord] = spectralVal;
                    rawValsCoord[iCoord] = rawVals[flatC(spatialCoord, spatialDims)];
                }
                
                // Write to file
                std::string dsetComponentName = dsetNameRaw+"_";
                for (auto component : componentIdx) {
                    dsetComponentName += std::to_string(component);
                }
                outfile << dsetComponentName+"_raw=" << rawValsCoord << std::endl;
                outfile << dsetComponentName+"_spectral=" << spectralValsCoord << std::endl;
            }            
        }
    }
}

} //END NAMESPACE HPP