#ifndef HPP_SPECTRAL_UTILS_H
#define HPP_SPECTRAL_UTILS_H

#include <mpi.h>
#include <cmath>
#include <vector>
#include <complex>
#include <fftw3-mpi.h>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <memory>
#include <stdint.h>
#include <string.h>
#include <hpp/config.h>
#include <hpp/hdfUtils.h>
#include <hpp/tensor.h>
#include <hpp/mpiUtils.h>
#include <hpp/simdUtils.h>
#include <hpp/stdlibReplacements.h>

namespace hpp
{

/**
 * @class FreeDelete
 * @author Michael Malahe
 * @date 08/03/17
 * @file spectralUtils.h
 * @brief A simple deleter for malloc'd memory
 */
struct FreeDelete {
    void operator()(void* x) {
        free(x);
    }
};

// FFTW THINGS //

/** FFTW uses row-major flattening
 *
*/
inline int fftwFlat2(int i0, int i1, int N0, int N1)
{
    return i0*N1 + i1;
}

inline int fftwFlat4(int i0, int i1, int i2, int i3, const std::vector<ptrdiff_t>& N)
{
    return i0*N[1]*N[2]*N[3] + i1*N[2]*N[3] + i2*N[3] + i3;
}

struct FFTWConfigRealND {
    MPI_Comm comm;
    size_t gridRank;
    std::vector<ptrdiff_t> realDims;
    std::vector<ptrdiff_t> realDimsPadded;
    std::vector<ptrdiff_t> complexDims;

    // Global number of points
    unsigned int NReal;
    unsigned int NComplex;

    // Local number of points
    ptrdiff_t localN0;
    ptrdiff_t local0Start;
    ptrdiff_t nLocalComplex;
    std::vector<ptrdiff_t> complexDimsLocal;
    std::vector<ptrdiff_t> realDimsLocal;
    ptrdiff_t nLocalReal;

    // Local number of points in memory
    ptrdiff_t nLocalComplexMem;
    ptrdiff_t nLocalRealPadded;
    std::vector<ptrdiff_t> realDimsPaddedLocal;

    // Local value arrays
    double *in;
    fftw_complex *out;
    double *backin;

    // Plans
    fftw_plan forwardPlan;
    fftw_plan backwardPlan;
};

FFTWConfigRealND prepareFFTWConfigRealND(const std::vector<ptrdiff_t>& realDims, MPI_Comm comm);
void destroyConfigRealND(FFTWConfigRealND& cfg);

inline double fftwComplexMag(fftw_complex comps)
{
    return std::sqrt(std::pow(comps[0], 2) + std::pow(comps[1], 2));
}

/**
 * @class SpectralDataset
 * @author Michael Malahe
 * @date 08/03/17
 * @file spectralUtils.h
 * @brief A dataset containing the truncated N-dimensional DFT of a scalar
 * @details The coordinates and coefficients are stored in aligned
 */
#define HPP_DEFAULT_ALIGNMENT 32
template <typename U>
class SpectralDataset
{
public:
    // Constructors
    SpectralDataset() : alignment(HPP_DEFAULT_ALIGNMENT) {
        nDims = 0;
        nTerms = 0;
    }
    SpectralDataset(const std::vector<std::vector<unsigned int>>& coordsList, const std::vector<std::complex<U>>& coeffList);

    // Getters
    const std::shared_ptr<U>& getCoeffs() const {
        return coeffs;
    }
    const std::shared_ptr<int>& getCoords() const {
        return coords;
    }
    unsigned int getNDims() const {
        return nDims;
    }
    unsigned int getNTerms() const {
        return nTerms;
    }

private:
    // Dataset dimensions
    unsigned int nDims;

    // Number of terms in the truncated DFT
    unsigned int nTerms;

    /** An array of the spectral coordinates of the terms,
     * arranged as [coord[0][0], coord[0][1], ..., coord[0][N-1],
     *              coord[1][0], coord[1][1], ..., coord[1][N-1],
     *              ...
     *              coord[M-1][0], coord[M-1][1], ..., coord[M-1][N-1]],
     * where M is the number of terms and N is the dimension.
     */
    std::shared_ptr<int> coords;

    /** The complex coefficients, arranged as
     * [Re(coeff[0]), Im(coeff[0]), ..., Re(coeff[M-1]), Im(coeff[M-1])]
     */
    std::shared_ptr<U> coeffs;

    // The byte alignment of the data
    unsigned int alignment = HPP_DEFAULT_ALIGNMENT;
};

/**
 * @class SpectralDatasetID
 * @author Michael Malahe
 * @date 28/03/17
 * @file spectralUtils.h
 * @brief A unique ID for a spectral dataset
 */
struct SpectralDatasetID {
    SpectralDatasetID(){;}
    
    // Constructors
    SpectralDatasetID(const std::string& baseName, const std::vector<unsigned int>& component) : baseName(baseName), component(component) {
        ;
    }
    explicit SpectralDatasetID(const std::string& baseName) : baseName(baseName) {
        ;
    }
    
    // Members
    std::string baseName;
    std::vector<unsigned int> component;
};

// Comparison for use with std::map
bool operator<(const SpectralDatasetID& l, const SpectralDatasetID& r);

// Operators defined for the sake of Boost Python vector indexing suite
bool operator==(const SpectralDatasetID& l, const SpectralDatasetID& r);

// Set names
#define HPP_DEFAULT_UNIFIED_COORDS_NAME "unified_coords"
#define HPP_DEFAULT_COORDS_SUFFIX "_coords"
#define HPP_DEFAULT_COEFFS_SUFFIX "_vals"

// Get derived names
inline std::string getCoordsName(std::string baseName) {
    return baseName+HPP_DEFAULT_COORDS_SUFFIX;
}
inline std::string getCoeffsName(std::string baseName) {
    return baseName+HPP_DEFAULT_COEFFS_SUFFIX;
}

template <typename U>
std::string getComponentSuffix(const std::vector<U>& componentIdx) {
    std::string suffix = "_";
    for(auto component : componentIdx) {
        suffix += std::to_string(component);
    }
    return suffix;
}
inline std::string getDefaultCoeffsDatasetName(SpectralDatasetID dsetID) {
    std::string dsetName;
    if (dsetID.component.size() == 0) {
        dsetName = dsetID.baseName;
    }
    else {        
        dsetName = dsetID.baseName+getComponentSuffix(dsetID.component);
    }
    return getCoeffsName(dsetName);
}

/**
 * @class SpectralDatabase
 * @author Michael Malahe
 * @date 07/03/17
 * @file spectralUtils.h
 * @brief Specifically for DFTs of real data
 * @details Currently assumes that database is square
 */

template <typename U>
class SpectralDatabase
{
public:
    // Default constructor
    SpectralDatabase() : alignment(HPP_DEFAULT_ALIGNMENT) {
        ;
    }
    
    // Construction from file
    SpectralDatabase(std::string dbFilename, std::vector<std::string> dsetBasenames, unsigned int nTerms, MPI_Comm comm, unsigned int refineMult=1);
    
    // Direct construction
    SpectralDatabase(std::vector<unsigned int> gd, std::vector<double> gs, std::vector<double> ge, std::map<SpectralDatasetID, SpectralDataset<U>> dsets);
    
    // IDFT
    U getIDFTReal(std::string dsetBasename, std::vector<unsigned int> componentIdx, std::vector<unsigned int> spatialCoord) const;
    U getIDFTReal(std::string dsetBasename, std::vector<unsigned int> spatialCoord) const;
    
    // Getters
    const std::shared_ptr<U>& getExpTable() const {
        return expTable;
    }
    const std::vector<unsigned int>& getGridDims() const {
        return gridDims;
    }
    const std::vector<double>& getGridEnds() const {
        return gridEnds;
    }
    const std::vector<double>& getGridLengths() const {
        return gridLengths;
    }
    const std::vector<double>& getGridStarts() const {
        return gridStarts;
    }
    const std::vector<double>& getGridSteps() const {
        return gridSteps;
    }
    const hsize_t& getNDims() const {
        return nDims;
    }
    SpectralDataset<U> getDataset(const SpectralDatasetID& dsetID) const {
        return dsets.at(dsetID);
    }
    unsigned int getNDsets() const {
        return dsets.size();
    }
    unsigned int getNTerms() const {
        return dsets.begin()->second.getNTerms();
    }

private:
    // Number of spatial dimensions
    hsize_t nDims = 0;

    // Grid dimensions
    std::vector<unsigned int> gridDims;

    // Grid spatial lengths
    std::vector<double> gridStarts;
    std::vector<double> gridEnds;
    std::vector<double> gridLengths;
    std::vector<double> gridSteps;

    // Datasets comprising the database
    std::map<SpectralDatasetID, SpectralDataset<U>> dsets;

    // Tables of exponentials
    std::shared_ptr<U> expTable;

    // Byte alignment for dynamically allocated memory
    unsigned int alignment = HPP_DEFAULT_ALIGNMENT;

    // Load dataset
    void loadDatasetSingleComponent(HDF5Handler& dbfile, std::string dsetBasename, std::vector<unsigned int> componentIdxUint, unsigned int nTerms, unsigned int refineMult=1);
    void loadDataset(HDF5Handler& dbfile, std::string dsetBasename, unsigned int nTerms, unsigned int refineMult=1);
    
    // Generate exponential table
    void generateExponentialTable();
};

/**
 * @class SpectralDatabaseUnified
 * @author Michael Malahe
 * @date 26/04/17
 * @file spectralUtils.h
 * @brief A spectral database for DFTs of real data.
 * @details This is specifically for "unified" coefficient datasets, which
 * here means that there is a single ordering of the dominant components that
 * all of the coefficients follow.
 */

template <typename U>
class SpectralDatabaseUnified
{
public:
    // Default constructor
    SpectralDatabaseUnified(){;}
    
    // Construction from file using given dataset IDs
    SpectralDatabaseUnified(std::string dbFilename, std::vector<SpectralDatasetID> dsetIDs, unsigned int nTerms, MPI_Comm comm, unsigned int refineMult=1);
    SpectralDatabaseUnified(std::string dbFilename, std::vector<SpectralDatasetID> dsetIDs, unsigned int nTerms, unsigned int refineMult=1);
    
    // Construction from file using discovered dataset IDs
    SpectralDatabaseUnified(std::string dbFilename, unsigned int nTerms, MPI_Comm comm, unsigned int refineMult=1);
    
    bool hasDset(SpectralDatasetID dsetID) {return coeffs.count(dsetID) > 0;}
    
    // IDFT
    U getIDFTReal(SpectralDatasetID dsetID, std::vector<unsigned int> spatialCoord) const;
    U getIDFTReal(std::string dsetBasename, std::vector<unsigned int> componentIdx, std::vector<unsigned int> spatialCoord) const;
    U getIDFTReal(std::string dsetFullname, std::vector<unsigned int> spatialCoord) const;
    
    // Getters
//    const std::shared_ptr<U>& getExpTable() const {
//        return expTable;
//    }
    const std::vector<unsigned int>& getGridDims() const {
        return gridDims;
    }
    const std::vector<double>& getGridEnds() const {
        return gridEnds;
    }
    const std::vector<double>& getGridLengths() const {
        return gridLengths;
    }
    const std::vector<double>& getGridStarts() const {
        return gridStarts;
    }
    const std::vector<double>& getGridSteps() const {
        return gridSteps;
    }
    const hsize_t& getNDims() const {
        return nDims;
    }
    std::shared_ptr<U> getCoeffs(const SpectralDatasetID& dsetID) const {
        return coeffs.at(dsetID);
    }
    unsigned int getNTerms() const {return nTerms;}
    const std::shared_ptr<int>& getCoords() const {return coords;}
private:
    //
    void constructFromHandler(HDF5Handler& dbFile, std::vector<SpectralDatasetID> dsetIDs, unsigned int nTerms, unsigned int refineMult=1);
    
    // Number of spatial dimensions
    hsize_t nDims = 0;
    
    // Number of terms
    unsigned int nTerms;

    // Grid dimensions
    std::vector<unsigned int> gridDims;

    // Grid spatial lengths
    std::vector<double> gridStarts;
    std::vector<double> gridEnds;
    std::vector<double> gridLengths;
    std::vector<double> gridSteps;

    // Coordinate ordering that applied to all sets of coefficients
    std::shared_ptr<int> coords;
    
    // The set of coefficients
    std::map<SpectralDatasetID, std::shared_ptr<U>> coeffs;

    // Tables of exponentials
    //std::shared_ptr<U> expTable;

    // Byte alignment for dynamically allocated memory
    const unsigned int alignment = HPP_DEFAULT_ALIGNMENT;

    // Load dataset
    void loadDatasets(HDF5Handler& dbfile, std::vector<SpectralDatasetID> dsetIDs, unsigned int nTerms, unsigned int refineMult=1);
    
    // Generate exponential table
    //void generateExponentialTable();
};

/**
 * @brief Load a spectral database
 * @param infile the file
 * @param dsetNameCoords the name of the dataset containing the spectral coordinates
 * @param dsetNameCoeffs the name of the dataset containing the corresponding spectral coefficients
 * @param coordsList the list of coordinates to return
 * @param coeffList the list of coefficients to return
 */
template <typename U>
void loadSpectralDatabase(hpp::HDF5Handler& infile, std::string dsetNameCoords, std::string dsetNameCoeffs, std::vector<hsize_t> componentIdx, std::vector<std::vector<unsigned int>>& coordsList, std::vector<std::complex<U>>& coeffList, unsigned int nTerms, unsigned int refineMult=1)
{
    // Property list for reading in data
    hid_t plist_in = infile.getPropertyListTransferIndependent();

    // Open datasets
    hid_t dsetCoords = infile.getDataset(dsetNameCoords);
    hid_t dsetCoeffs = infile.getDataset(dsetNameCoeffs);

    // Get dimensions
    std::vector<hsize_t> dsetCoordsDims = hpp::getDatasetDims(dsetCoords);
    std::vector<hsize_t> dsetCoeffsDims = hpp::getDatasetDims(dsetCoeffs);
    unsigned int nCoeffs = dsetCoordsDims.end()[-2];
    unsigned int nDims = dsetCoordsDims.back();
    if (nCoeffs != dsetCoeffsDims.back()) throw std::runtime_error("Mismatch in dimensions of coeffs and coords.");
    if (nTerms > nCoeffs) throw std::runtime_error("Requested more terms than exist in the dataset.");

    // Grid information
    hid_t dsetGridDims = infile.getDataset("grid_dims");
    std::vector<hsize_t> nDimsArray = hpp::getDatasetDims(dsetGridDims);
    std::vector<unsigned int> gridDims(nDims);
    std::vector<unsigned short int> gridDimsBuffer(nDims);
    hpp::readSingleHDF5Array(dsetGridDims, plist_in, nDimsArray, gridDimsBuffer.data());
    std::copy(gridDimsBuffer.begin(), gridDimsBuffer.end(), gridDims.data());

    // Read in raw data
    std::vector<unsigned short int> coordsBuffer(nTerms*nDims);
    std::vector<hpp::hdf_complex_t> coeffsBuffer(nTerms);

    hpp::HDFReadWriteParamsC coordsReadParams;
    coordsReadParams.dataOffset = componentIdx;
    coordsReadParams.dataOffset.push_back(0);
    coordsReadParams.dataOffset.push_back(0);
    coordsReadParams.dataCount = std::vector<hsize_t>(componentIdx.size(), 1);
    coordsReadParams.dataCount.push_back(nTerms);
    coordsReadParams.dataCount.push_back(nDims);
    coordsReadParams.datatype = H5Dget_type(dsetCoords);
    hpp::readHDF5SimpleArray(dsetCoords, plist_in, coordsReadParams, coordsBuffer.data());

    hpp::HDFReadWriteParamsC coeffsReadParams;
    coeffsReadParams.dataOffset = componentIdx;
    coeffsReadParams.dataOffset.push_back(0);
    coeffsReadParams.dataCount = std::vector<hsize_t>(componentIdx.size(), 1);
    coeffsReadParams.dataCount.push_back(nTerms);
    coeffsReadParams.datatype = H5Dget_type(dsetCoeffs);
    hpp::readHDF5SimpleArray(dsetCoeffs, plist_in, coeffsReadParams, coeffsBuffer.data());

    // Spectral refinement
    if (refineMult == 0) {
        throw std::runtime_error("Refinement multiplier must be strictly positive.");
    } else {
        for (auto&& dim : gridDims) {
            dim *= refineMult;
        }
    }

    // Convert coords and coeffs
    coordsList.resize(nTerms);
    coeffList.resize(nTerms);

    // The number of actual numerical terms is less than the raw number requested,
    // because using the fact that the data is real, we can simply double
    // the terms that are supposed to have their conjugates included
    unsigned int nTermsIncluded = 0; //number of analytic terms that have been included
    unsigned int nTermsNumerical = 0; //number of numerical terms required to capture their effect
    for (unsigned int i=0; nTermsIncluded<nTerms; i++) {
        // Coords
        coordsList[i].resize(nDims);
        unsigned short int *start = coordsBuffer.data()+i*nDims;
        unsigned short int *end = start+nDims;//Note that "end" is the address one *after* the final element
        std::copy(start, end, coordsList[i].data());

        // Spectral refinement
        for (auto&& coord : coordsList[i]) {
            coord *= refineMult;
        }

        // Coeffs
        hpp::hdf_complex_t rawVal = coeffsBuffer[i];
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
            if (nTermsIncluded<nTerms) {
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
}

/**
 * @brief TruncatedIFFTND where all dimensions have the same number of points
 * @param expTable table of pre-computed complex exponentials \f$ [e^{0}, e^{2\pi i (1/N)}, e^{2\pi i (2/N)}, ..., e^{2\pi i (N-1)/N}}] \f$
 * @param nTerms number of terms
 * @return
 */
template <typename U>
std::complex<U> TruncatedIFFTNDSingleSquare(const unsigned int spatialDim, const std::vector<std::complex<U>>& expTable, const std::vector<std::vector<unsigned int>>& spectralCoordsList, const std::vector<std::complex<U>>& coeffs, std::vector<unsigned int> spatialCoord, unsigned int nTerms)
{
    // Size checks
    unsigned int nDims = spatialCoord.size();
    unsigned int nCoeffs = spectralCoordsList.size();
    if (coeffs.size() != nCoeffs) throw std::runtime_error("Mismatch between number of coordinates and coefficients.");
    if (nTerms > nCoeffs) {
        std::string errString = "Requested more terms than have been loaded: ";
        errString += std::to_string(nTerms) + " > " + std::to_string(nCoeffs);
        throw std::runtime_error(errString);
    }

    // Restore
    std::complex<U> val = 0.0;
    std::complex<U> iUnit(0, 1);
    std::complex<U> seriesTerm;
    for (unsigned int iCoeff=0; iCoeff<nTerms; iCoeff++) {
        // Add terms
        unsigned int expFactorInt = 0;
        for (unsigned int iDim=0; iDim<nDims; iDim++) {
            expFactorInt += spectralCoordsList[iCoeff][iDim]*spatialCoord[iDim];
        }

        // Range reduce inputs
        expFactorInt = expFactorInt % spatialDim;
        seriesTerm = coeffs[iCoeff]*expTable[expFactorInt];

        // Add actual terms
        val += seriesTerm;
    }

    // Return
    return val;
}

template <typename U>
U TruncatedIFFTNDSingleSquareReal(const unsigned int spatialDim, const std::vector<std::complex<U>>& expTable, const std::vector<std::vector<unsigned int>>& spectralCoordsList, const std::vector<std::complex<U>>& coeffs, std::vector<unsigned int> spatialCoord, unsigned int nTerms)
{
    // Size checks
    unsigned int nDims = spatialCoord.size();
    unsigned int nCoeffs = spectralCoordsList.size();
    if (coeffs.size() != nCoeffs) throw std::runtime_error("Mismatch between number of coordinates and coefficients.");
    if (nTerms > nCoeffs) throw std::runtime_error("Requested more terms than have been loaded.");

    // Restore
    U val = 0.0;
    std::complex<U> coeff;
    std::complex<U> expVal;
    for (unsigned int iCoeff=0; iCoeff<nTerms; iCoeff++) {
        // Add terms
        unsigned int expFactorInt = 0;
        for (unsigned int iDim=0; iDim<nDims; iDim++) {
            expFactorInt += spectralCoordsList[iCoeff][iDim]*spatialCoord[iDim];
        }

        // Range reduce inputs
        expFactorInt = expFactorInt % spatialDim;

        // Operands
        coeff = coeffs[iCoeff];
        expVal = expTable[expFactorInt];
        U seriesTerm = std::real(coeff)*std::real(expVal) - std::imag(coeff)*std::imag(expVal);

        // Add actual terms
        val += seriesTerm;
    }

    // Return
    return val;
}

/**
 * @brief Scalar implementation of hpp.TruncatedIFFTNDSingleSquareReal
 * @details No guarantees that the compiler won't use vector instructions.
 */
template <typename U>
U TruncatedIFFTNDSingleSquareRealScalar(const unsigned int nDims, const unsigned int spatialDim, const U expTable[], const int spectralCoordsList[], const U coeffs[], std::vector<unsigned int> spatialCoord, unsigned int nTerms)
{
    U val = 0.0;
    for (unsigned int iCoeff=0; iCoeff<nTerms; iCoeff++) {
        // Add terms
        int expFactorInt = 0;
        for (unsigned int iDim=0; iDim<nDims; iDim++) {
            expFactorInt += spectralCoordsList[nDims*iCoeff+iDim]*spatialCoord[iDim];
        }

        // Operands
        U seriesTerm = coeffs[2*iCoeff]*expTable[2*expFactorInt] - coeffs[2*iCoeff+1]*expTable[2*expFactorInt+1];

        // Add actual terms
        val += seriesTerm;
    }

    // Return
    return val;
}

// Without a lookup table of complex exponentials
// Scalar processor implementation
template <typename U>
U TruncatedIFFTNDSingleSquareRealScalar(const unsigned int nDims, const unsigned int spatialDim, const int spectralCoordsList[], const U coeffs[], std::vector<unsigned int> spatialCoord, unsigned int nTerms)
{
    U val = 0.0;
    for (unsigned int iCoeff=0; iCoeff<nTerms; iCoeff++) {
        // Add terms
        int expFactorInt = 0;
        for (unsigned int iDim=0; iDim<nDims; iDim++) {
            expFactorInt += spectralCoordsList[nDims*iCoeff+iDim]*spatialCoord[iDim];
        }
        
        // Range reduce
        expFactorInt = expFactorInt % spatialDim;
        
        // Evaluate term    
        U expValRe = std::cos(2*expFactorInt*M_PI/spatialDim);
        U expValIm = std::sin(2*expFactorInt*M_PI/spatialDim);
        U seriesTerm = coeffs[2*iCoeff]*expValRe - coeffs[2*iCoeff+1]*expValIm;

        // Add actual terms
        val += seriesTerm;
    }

    // Return
    return val;
}

/**
 * @brief AVX implementation of hpp.TruncatedIFFT4DSingleSquareReal
 */
#ifdef HPP_USE_AVX
template <typename U>
U TruncatedIFFT4DSingleSquareRealAVX(const unsigned int spatialDim, const U expTable[], const int spectralCoordsList[], const U coeffs[], std::vector<unsigned int> spatialCoord, unsigned int nTerms)
{
    __m256 productsSumVecF = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    __m256d productsSumVecD = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m128i spatialCoordVec = _mm_set_epi32((int)spatialCoord[3],(int)spatialCoord[2],(int)spatialCoord[1],(int)spatialCoord[0]);
    __m128i twos = _mm_set_epi32(2,2,2,2);
    unsigned int nChunks = nTerms/4;
    for (unsigned int iChunk=0; iChunk<nChunks; iChunk++) {
        const unsigned int iCoeff0 = iChunk*4;
        const unsigned int iCoeff1 = iChunk*4+1;
        const unsigned int iCoeff2 = iChunk*4+2;
        const unsigned int iCoeff3 = iChunk*4+3;

        // Products of spectral and spatial coordinates
        __m128i coordsVec  = _mm_load_si128((__m128i*)(spectralCoordsList+4*iCoeff0));
        // coordsVec = [coords[iCoeff0][0],coords[iCoeff0][1],coords[iCoeff0][2],coords[iCoeff0][3]]
        __m128i coordProdsVec0 = _mm_mullo_epi32(coordsVec, spatialCoordVec);
        // coordProdsVec0 = [coords[iCoeff0][0]*s[0],coords[iCoeff0][1]*s[1],coords[iCoeff0][2]*s[2],coords[iCoeff0][3]*s[3]]

        coordsVec = _mm_load_si128((__m128i*)(spectralCoordsList+4*iCoeff1));
        __m128i coordProdsVec1 = _mm_mullo_epi32(coordsVec, spatialCoordVec);

        coordsVec = _mm_load_si128((__m128i*)(spectralCoordsList+4*iCoeff2));
        __m128i coordProdsVec2 = _mm_mullo_epi32(coordsVec, spatialCoordVec);

        coordsVec = _mm_load_si128((__m128i*)(spectralCoordsList+4*iCoeff3));
        __m128i coordProdsVec3 = _mm_mullo_epi32(coordsVec, spatialCoordVec);

        // Combined
        __m128i addedPairsVec01 = _mm_hadd_epi32(coordProdsVec0, coordProdsVec1);
        // addedPairsVec01 =
        // [coords[iCoeff0][0]*s[0]+coords[iCoeff0][1]*s[1],
        //  coords[iCoeff0][2]*s[2]+coords[iCoeff0][3]*s[3],
        //  coords[iCoeff1][0]*s[0]+coords[iCoeff1][1]*s[1],
        //  coords[iCoeff1][2]*s[2]+coords[iCoeff1][3]*s[3]]
        __m128i addedPairsVec23 = _mm_hadd_epi32(coordProdsVec2, coordProdsVec3);
        // addedPairsVec23 =
        // [coords[iCoeff2][0]*s[0]+coords[iCoeff2][1]*s[1],
        //  coords[iCoeff2][2]*s[2]+coords[iCoeff2][3]*s[3],
        //  coords[iCoeff3][0]*s[0]+coords[iCoeff3][1]*s[1],
        //  coords[iCoeff3][2]*s[2]+coords[iCoeff3][3]*s[3]]
        __m128i fullyAddedVec0123 = _mm_hadd_epi32(addedPairsVec01, addedPairsVec23);
        // fullyAddedVec0123 =
        // [coords[iCoeff0][0]*s[0]+coords[iCoeff0][1]*s[1]+coords[iCoeff0][2]*s[2]+coords[iCoeff0][3]*s[3],
        //  coords[iCoeff1][0]*s[0]+coords[iCoeff1][1]*s[1]+coords[iCoeff1][2]*s[2]+coords[iCoeff1][3]*s[3],
        //  coords[iCoeff2][0]*s[0]+coords[iCoeff2][1]*s[1]+coords[iCoeff2][2]*s[2]+coords[iCoeff2][3]*s[3],
        //  coords[iCoeff3][0]*s[0]+coords[iCoeff3][1]*s[1]+coords[iCoeff3][2]*s[2]+coords[iCoeff3][3]*s[3]]
        fullyAddedVec0123 = _mm_mullo_epi32(fullyAddedVec0123, twos);
        int32_t *expFac = (int32_t*)&fullyAddedVec0123;

        // Add up products
        // FLOAT
        if (std::is_same<U, float>::value) {
            __m256 coeffVecF = _mm256_load_ps((float*)coeffs+2*iCoeff0);
            __m256 expValVecF = _mm256_loadu4_m64((float*)expTable+expFac[3], (float*)expTable+expFac[2], (float*)expTable+expFac[1], (float*)expTable+expFac[0]);
            __m256 productsVecF = _mm256_mul_ps(coeffVecF, expValVecF);
            productsSumVecF = _mm256_add_ps(productsSumVecF, productsVecF);
            // DOUBLE
        } else if (std::is_same<U, double>::value) {
            // First set
            __m256d coeffVecD = _mm256_load_pd((double*)coeffs+2*iCoeff0);
            __m256d expValVecD = _mm256_loadu2_m128d((double*)expTable+expFac[1], (double*)expTable+expFac[0]);
            __m256d productsVecD = _mm256_mul_pd(coeffVecD, expValVecD);
            productsSumVecD = _mm256_add_pd(productsSumVecD, productsVecD);

            // Second set
            coeffVecD = _mm256_load_pd((double*)coeffs+2*iCoeff2);
            expValVecD = _mm256_loadu2_m128d((double*)expTable+expFac[3], (double*)expTable+expFac[2]);
            productsVecD = _mm256_mul_pd(coeffVecD, expValVecD);
            productsSumVecD = _mm256_add_pd(productsSumVecD, productsVecD);
        }
    }

    // Return
    // FLOAT
    if (std::is_same<U, float>::value) {
        float *productsSum = (float*)&productsSumVecF;
        return productsSum[0] - productsSum[1] + productsSum[2] - productsSum[3] + productsSum[4] - productsSum[5] + productsSum[6] - productsSum[7];
        // DOUBLE
    } else if (std::is_same<U, double>::value) {
        double *productsSum = (double*)&productsSumVecD;
        return productsSum[0] - productsSum[1] + productsSum[2] - productsSum[3];
    }
}
#endif /*HPP_USE_AVX*/

/**
 * @brief AVX2 implementation of hpp.TruncatedIFFT4DSingleSquareReal
 */
#ifdef HPP_USE_AVX2
template <typename U>
U TruncatedIFFT4DSingleSquareRealAVX2(const unsigned int spatialDim, const U expTable[], const int spectralCoordsList[], const U coeffs[], std::vector<unsigned int> spatialCoord, unsigned int nTerms)
{
    __m256 productsSumVecF = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    __m256d productsSumVecD = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256i spatialCoordVec = _mm256_set_epi32((int)spatialCoord[3],(int)spatialCoord[2],(int)spatialCoord[1],(int)spatialCoord[0],(int)spatialCoord[3],(int)spatialCoord[2],(int)spatialCoord[1],(int)spatialCoord[0]);
    __m256i twos = _mm256_set_epi32(2,2,2,2,2,2,2,2);
    unsigned int nChunks = nTerms/8;
    for (unsigned int iChunk=0; iChunk<nChunks; iChunk++) {
        const unsigned int iCoeff0 = iChunk*8;
        const unsigned int iCoeff2 = iChunk*8+2;
        const unsigned int iCoeff4 = iChunk*8+4;
        const unsigned int iCoeff6 = iChunk*8+6;

        // Products of spectral and spatial coordinates
        __m256i coordsVec  = _mm256_load_si256((__m256i*)(spectralCoordsList+4*iCoeff0));
        // coordsVec =
        // [coords[iCoeff0][0],coords[iCoeff0][1],coords[iCoeff0][2],coords[iCoeff0][3],
        //  coords[iCoeff1][0],coords[iCoeff1][1],coords[iCoeff1][2],coords[iCoeff1][3]]
        __m256i coordProdsVec01 = _mm256_mullo_epi32(coordsVec, spatialCoordVec);
        // coordProdsVec01 =
        // [coords[iCoeff0][0]*s[0],coords[iCoeff0][1]*s[1],coords[iCoeff0][2]*s[2],coords[iCoeff0][3]*s[3],
        //  coords[iCoeff1][0]*s[0],coords[iCoeff1][1]*s[1],coords[iCoeff1][2]*s[2],coords[iCoeff1][3]*s[3]]
        // Combined
        coordsVec  = _mm256_load_si256((__m256i*)(spectralCoordsList+4*iCoeff2));
        __m256i coordProdsVec23 = _mm256_mullo_epi32(coordsVec, spatialCoordVec);

        coordsVec  = _mm256_load_si256((__m256i*)(spectralCoordsList+4*iCoeff4));
        __m256i coordProdsVec45 = _mm256_mullo_epi32(coordsVec, spatialCoordVec);

        coordsVec  = _mm256_load_si256((__m256i*)(spectralCoordsList+4*iCoeff6));
        __m256i coordProdsVec67 = _mm256_mullo_epi32(coordsVec, spatialCoordVec);

        // Note that the ordering produced by _mm256_hadd_epi32 is awkward. The explanation below, though bizarre-looking, is correct
        __m256i addedPairsVec0213 = _mm256_hadd_epi32(coordProdsVec01, coordProdsVec23);
        // addedPairsVec0213 =
        // coords[iCoeff0][0]*s[0]+coords[iCoeff0][1]*s[1], coords[iCoeff0][2]*s[2]+coords[iCoeff0][3]*s[3],
        // coords[iCoeff2][0]*s[0]+coords[iCoeff2][1]*s[1], coords[iCoeff2][2]*s[2]+coords[iCoeff2][3]*s[3],
        // coords[iCoeff1][0]*s[0]+coords[iCoeff1][1]*s[1], coords[iCoeff1][2]*s[2]+coords[iCoeff1][3]*s[3],
        // coords[iCoeff3][0]*s[0]+coords[iCoeff3][1]*s[1], coords[iCoeff3][2]*s[2]+coords[iCoeff3][3]*s[3],
        __m256i addedPairsVec4657 = _mm256_hadd_epi32(coordProdsVec45, coordProdsVec67);
        __m256i fullyAddedVec02461357 = _mm256_hadd_epi32(addedPairsVec0213, addedPairsVec4657);
        // The mapping from logical indices to array indices is now
        // 0->0
        // 1->4
        // 2->1
        // 3->5
        // 4->2
        // 5->6
        // 6->3
        // 7->7
        fullyAddedVec02461357 = _mm256_mullo_epi32(fullyAddedVec02461357, twos);
        int32_t *expFac = (int32_t*)&fullyAddedVec02461357;

        // Add up products
        // FLOAT
        if (std::is_same<U, float>::value) {
            // First set
            __m256 coeffVecF = _mm256_load_ps((float*)coeffs+2*iCoeff0);
            __m256 expValVecF = _mm256_loadu4_m64((float*)expTable+expFac[5], (float*)expTable+expFac[1], (float*)expTable+expFac[4], (float*)expTable+expFac[0]);
            __m256 productsVecF = _mm256_mul_ps(coeffVecF, expValVecF);
            productsSumVecF = _mm256_add_ps(productsSumVecF, productsVecF);

            // Second set
            coeffVecF = _mm256_load_ps((float*)coeffs+2*iCoeff4);
            expValVecF = _mm256_loadu4_m64((float*)expTable+expFac[7], (float*)expTable+expFac[3], (float*)expTable+expFac[6], (float*)expTable+expFac[2]);
            productsVecF = _mm256_mul_ps(coeffVecF, expValVecF);
            productsSumVecF = _mm256_add_ps(productsSumVecF, productsVecF);
            // DOUBLE
        } else if (std::is_same<U, double>::value) {
            // First set
            __m256d coeffVecD = _mm256_load_pd((double*)coeffs+2*iCoeff0);
            __m256d expValVecD = _mm256_loadu2_m128d((double*)expTable+expFac[4], (double*)expTable+expFac[0]);
            __m256d productsVecD = _mm256_mul_pd(coeffVecD, expValVecD);
            productsSumVecD = _mm256_add_pd(productsSumVecD, productsVecD);

            // Second set
            coeffVecD = _mm256_load_pd((double*)coeffs+2*iCoeff2);
            expValVecD = _mm256_loadu2_m128d((double*)expTable+expFac[5], (double*)expTable+expFac[1]);
            productsVecD = _mm256_mul_pd(coeffVecD, expValVecD);
            productsSumVecD = _mm256_add_pd(productsSumVecD, productsVecD);

            // Third set
            coeffVecD = _mm256_load_pd((double*)coeffs+2*iCoeff4);
            expValVecD = _mm256_loadu2_m128d((double*)expTable+expFac[6], (double*)expTable+expFac[2]);
            productsVecD = _mm256_mul_pd(coeffVecD, expValVecD);
            productsSumVecD = _mm256_add_pd(productsSumVecD, productsVecD);

            // Fourth set
            coeffVecD = _mm256_load_pd((double*)coeffs+2*iCoeff6);
            expValVecD = _mm256_loadu2_m128d((double*)expTable+expFac[7], (double*)expTable+expFac[3]);
            productsVecD = _mm256_mul_pd(coeffVecD, expValVecD);
            productsSumVecD = _mm256_add_pd(productsSumVecD, productsVecD);
        }
    }

    // Return
    // FLOAT
    if (std::is_same<U, float>::value) {
        float *productsSum = (float*)&productsSumVecF;
        return productsSum[0] - productsSum[1] + productsSum[2] - productsSum[3] + productsSum[4] - productsSum[5] + productsSum[6] - productsSum[7];
        // DOUBLE
    } else if (std::is_same<U, double>::value) {
        double *productsSum = (double*)&productsSumVecD;
        return productsSum[0] - productsSum[1] + productsSum[2] - productsSum[3];
    }
}
#endif /*HPP_USE_AVX2*/

/**
 * @brief
 * @param spatialDim
 * @param expTable 16/32-byte (float/double) aligned array of real-im pairs from the exp table
 * @param coeffs 16/32-byte (float/double) aligned array of real-im pairs from the coeffs list
 * @param spatialCoord
 * @param nTerms
 * @return
 */
template <typename U>
U TruncatedIFFT4DSingleSquareReal(const unsigned int spatialDim, const U expTable[], const int spectralCoordsList[], const U coeffs[], const std::vector<unsigned int>& spatialCoord, unsigned int nTerms)
{
    // Architecture-specific implementations
#if defined HPP_USE_AVX2
    return TruncatedIFFT4DSingleSquareRealAVX2(spatialDim, expTable, spectralCoordsList, coeffs, spatialCoord, nTerms);
#elif defined HPP_USE_AVX
    return TruncatedIFFT4DSingleSquareRealAVX(spatialDim, expTable, spectralCoordsList, coeffs, spatialCoord, nTerms);
#else
    return TruncatedIFFTNDSingleSquareRealScalar(4, spatialDim, expTable, spectralCoordsList, coeffs, spatialCoord, nTerms);
#endif
}

template <typename U>
U TruncatedIFFTNDSingleSquareReal(const unsigned int nDims, const unsigned int spatialDim, const U expTable[], const int spectralCoordsList[], const U coeffs[], const std::vector<unsigned int>& spatialCoord, unsigned int nTerms)
{
    // Architecture-specific implementations
    /// @todo CPU SIMD implementations for arbitrary dimension
    return TruncatedIFFTNDSingleSquareRealScalar(nDims, spatialDim, expTable, spectralCoordsList, coeffs, spatialCoord, nTerms);
}

template <typename U>
U TruncatedIFFT4DSingleSquareReal(const unsigned int spatialDim, const int spectralCoordsList[], const U coeffs[], const std::vector<unsigned int>& spatialCoord, unsigned int nTerms)
{
    // Architecture-specific implementations
    /// @todo CPU SIMD implementations for arbitrary dimension
    return TruncatedIFFTNDSingleSquareRealScalar(4, spatialDim, spectralCoordsList, coeffs, spatialCoord, nTerms);
}

template <typename U>
U TruncatedIFFTNDSingleSquareReal(const unsigned int nDims, const unsigned int spatialDim, const int spectralCoordsList[], const U coeffs[], const std::vector<unsigned int>& spatialCoord, unsigned int nTerms)
{
    // Architecture-specific implementations
    /// @todo CPU SIMD implementations for arbitrary dimension
    return TruncatedIFFTNDSingleSquareRealScalar(nDims, spatialDim, spectralCoordsList, coeffs, spatialCoord, nTerms);
}

void evaluateSpectralCompressionErrorFull(std::string rawDBName, std::string spectralDBName, std::string errorDBName, unsigned int nTermsMax, std::string outFilename, MPI_Comm comm);

void evaluateSpectralCompressionErrorAxisSlice(std::string rawDBName, std::string spectralDBName, std::vector<std::string> dsetBasenames, unsigned int nTermsMax, unsigned int refineMult, std::vector<int> axisSlice, std::string outFilename, MPI_Comm comm);

void evaluateSpectralCompressionErrorFullUnified(std::string rawDBName, std::string spectralDBName, std::string errorDBName, unsigned int nTermsMax, std::string outFilename, MPI_Comm comm);

void evaluateSpectralCompressionErrorAxisSliceUnified(std::string rawDBName, std::string spectralDBName, std::vector<std::string> dsetBasenames, unsigned int nTermsMax, unsigned int refineMult, std::vector<int> axisSlice, std::string outFilename, MPI_Comm comm);


} //END NAMESPACE HPP

#endif /* HPP_SPECTRAL_UTILS_H */
