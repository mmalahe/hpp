/** @file spectralUtilsCUDA.h
* @author Michael Malahe
* @brief Header file for spectral utilities CUDA implementations
* @details
*/

#ifndef HPP_SPECTRAL_UTILS_CUDA_H
#define HPP_SPECTRAL_UTILS_CUDA_H

#include <hpp/config.h>
#include <hpp/crystal.h>
#include <hpp/cudaUtils.h>

namespace hpp
{

#ifdef HPP_USE_CUDA
/**
 * @class SpectralData
 * @author Michael Malahe
 * @date 04/04/17
 * @file spectralUtilsCUDA.h
 * @brief 
 * @tparam T the scalar datatype
 * @tparam N the dimension of the database
 */
template<typename T, unsigned int N>
struct SpectralDataCUDA {
    unsigned int coords[N];
    T coeff[2];
};

template<unsigned int N>
class SpectralCoordCUDA {
public:
    __host__ __device__ SpectralCoordCUDA(){;}
    __host__ __device__ unsigned int getVal(const unsigned int i) const {
        return data[i];
    }
    __host__ __device__ unsigned int& operator()(const unsigned int i) {
        return data[i];
    }
    
private:
    unsigned int data[N];
};

template<typename T>
struct SpectralCoeffCUDA {
    T re;
    T im;
};

/**
 * @class SpectralDatasetCUDA
 * @author Michael Malahe
 * @date 12/04/17
 * @file spectralUtilsCUDA.h
 * @brief
 * @tparam T the scalar datatype
 * @tparam N the dimension of the database 
 */
template<typename T, unsigned int N>
struct SpectralDatasetCUDA {
    SpectralCoeffCUDA<T> *coeffs;
    SpectralCoordCUDA<N> *coords;
    unsigned int nTerms;
};

/**
 * @class SpectralDatabaseCUDA
 * @author Michael Malahe
 * @date 04/04/17
 * @file spectralUtilsCUDA.h
 * @brief A spectral database for DFTs of real data.
 * @details Instances of the class itself may live on the device or host. However, all of the dynamically-allocated memory is on the device.
 * The approach to dynamic memory has two parts. First, there are raw pointers to device memory. 
 * These are needed for use in the device member functions. Second, each of these raw pointers is additionally wrapped in a shared_ptr.
 * This ensures that the memory is freed upon destruction of the final holder of the memory, and also that the database can be passed around and copied freely.
 * 
 * The class is designed around having its getIDFTReal members be as blazing
 * fast as possible. To this end, there are a number of awkward but worthwhile
 * optimisations.
 * 1. The sign on the imaginary terms of the Fourier coefficients is pre-negated 
 * in the constructor. This allows the exp*coeff multiplication to be done in
 * exactly two FMA operations.
 * @tparam T the scalar datatype
 * @tparam N the dimension of the database
 */
template<typename T, unsigned int N>
class SpectralDatabaseCUDA
{
public:
    SpectralDatabaseCUDA();
    SpectralDatabaseCUDA(const SpectralDatabase<T>& dbIn, const std::vector<SpectralDatasetID>& dsetIDs);
    __device__ T getIDFTRealD(unsigned int dsetIdx, unsigned int *spatialCoord) const;
    __device__ T getIDFTRealDShared(unsigned int dsetIdx, unsigned int *spatialCoord, unsigned int nShared, SpectralCoordCUDA<N> *sharedCoords, SpectralCoeffCUDA<T> *sharedCoeffs) const;
    T getIDFTRealH(unsigned int dsetIdx, std::vector<unsigned int> spatialCoord) const;
    
    // Getters
    __device__ T* getGridStarts() {return gridStarts;}
    __device__ T* getGridSteps() {return gridSteps;}
    __device__ unsigned int * getGridDims() {return gridDims;}
    unsigned int getNDsets() const {return nDsets;}
    unsigned int getNTermsTypical() const {return nTermsTypical;}
    
protected:
    
private:    
    // Grid dimensions    
    unsigned int *gridDims;
    std::shared_ptr<unsigned int> gridDimsSharedPtr;
    
    // Grid spatial parameters    
    T *gridStarts;
    std::shared_ptr<T> gridStartsSharedPtr;    
    T *gridSteps;
    std::shared_ptr<T> gridStepsSharedPtr;
    
    // Number of datasets
    unsigned int nDsets;
    
    // Number of terms in a typical dataset
    unsigned int nTermsTypical;

    // Spectral data
    SpectralDatasetCUDA<T,N> *dsets;
    
    // Shared pointers to assure correct copying and destruction
    std::shared_ptr<SpectralDatasetCUDA<T,N>> dsetsSharedPtr;
    std::vector<std::shared_ptr<SpectralCoordCUDA<N>>> coordSharedPtrs;
    std::vector<std::shared_ptr<SpectralCoeffCUDA<T>>> coeffSharedPtrs;    
};

// Device IDFT
template <typename T, unsigned int N>
__device__ T SpectralDatabaseCUDA<T,N>::getIDFTRealD(unsigned int dsetIdx, unsigned int *spatialCoord) const {
    // Get correct dataset
    SpectralDatasetCUDA<T,N> dset = dsets[dsetIdx];
    
    // Initialise value
    T val = 0;
    
    // Exponential argument common factor
    T expArgFactor = 2*((T)M_PI)/gridDims[0];
    
    // Add terms
    for (unsigned int i=0; i<dset.nTerms; i++) {
        // Get exponential index
        unsigned int expInd = 0;
        for (unsigned int j=0; j<N; j++) {
            expInd += spatialCoord[j]*dset.coords[i](j);
        }
        
        // Range reduce exponential index. This is a significant (~20%) saving 
        // compared with having the intrinsic trig functions do it. 
        expInd = expInd&(gridDims[0]-1);// Optimisation for gridDims[0] a power of two
        
        // Get complex exponential
        T expArg = expInd*expArgFactor;
        T expVal[2];
        sincosIntrinsic(expArg, &(expVal[1]), &(expVal[0]));
        
        // Add real part of term
        val = fmaIntrinsic(dset.coeffs[i].re, expVal[0], val);
        val = fmaIntrinsic(dset.coeffs[i].im, expVal[1], val);
    }
    
    // Return
    return val;
}

/**
 * @brief Device IDFTD
 * @details The grid dimension must be a power of two.  For performance reasons 
 * this is not checked in this function. The particular optimisation is to
 * replace the modulo in the range reduction step with the following bitwise
 * operation trick for powers of two. Suppose we want y%x, and x is a power of
 * two: \f$x = 2^z\f$. The binary representation of x has the digit 1 in the z position
 * and zeroes elsewhere. Now \f$ x - 1 = 2^z - 1 \f$ has a 0 in the z position and a 1
 * in every lower bit. If we then take y&(x-1), all of the bits of y above and 
 * including the z position represent values that are divisible by x, and are
 * accordingly zeroed in the result. All of the bits below the z position are
 * left intact (being bitwise & with all 1's), leaving us with the remainder.
 * @param dsetIdx
 * @param spatialCoord
 * @param nShared
 * @param sharedCoords
 * @param sharedCoeffs
 */
template <typename T, unsigned int N>
__device__ T SpectralDatabaseCUDA<T,N>::getIDFTRealDShared(unsigned int dsetIdx, unsigned int *spatialCoord, unsigned int nShared, SpectralCoordCUDA<N> *sharedCoords, SpectralCoeffCUDA<T> *sharedCoeffs) const {
    // Get correct dataset
    SpectralDatasetCUDA<T,N> dset = dsets[dsetIdx];
    
    // Dataset size
    unsigned int nTerms = dset.nTerms;
    
    // Initialise value
    T val = 0;
    
    // Exponential argument common factor
    T expArgFactor = 2*((T)M_PI)/gridDims[0];
    
    // Read into shared memory as a block
    // termsPerRead is the number of shared values that are read per large read
    // nReads is the number of times that we must perform a large read from
    // global memory to shared memory.
    unsigned int termsPerLargeRead = nShared;
    unsigned int nReads = nTerms/termsPerLargeRead;
    if (nTerms % termsPerLargeRead != 0) nReads++;
    
    // termsPerBlockRead is the number of values that are read when every thread 
    // in the block does a single read
    unsigned int termsPerBlockRead = blockDim.x;
    
    // Loop by shared memory reads
    for (unsigned int iRead=0; iRead<nReads; iRead++) {
        // Read data into shared memory
        unsigned int readStartGlobal = iRead*termsPerLargeRead+threadIdx.x;
        unsigned int readEndGlobal = umin((iRead+1)*termsPerLargeRead, nTerms);
        for (unsigned int readIdxGlobal = readStartGlobal; readIdxGlobal<readEndGlobal; readIdxGlobal+=termsPerBlockRead) {
            unsigned int readIdxShared = readIdxGlobal%termsPerLargeRead;
            sharedCoords[readIdxShared] = dset.coords[readIdxGlobal];
            sharedCoeffs[readIdxShared] = dset.coeffs[readIdxGlobal];
        }
        
        // Sync after read
        __syncthreads();
        
        // Compute available terms
        unsigned int termsStartGlobal = iRead*termsPerLargeRead;
        unsigned int termsEndGlobal = umin(termsStartGlobal+termsPerLargeRead, nTerms);
        unsigned int termsStartShared = 0;
        unsigned int termsEndShared = termsEndGlobal-termsStartGlobal;
        for (unsigned int i=termsStartShared; i<termsEndShared; i++) {
            // Get exponential index
            unsigned int expInd = spatialCoord[0]*sharedCoords[i](0);
            for (unsigned int j=1; j<N; j++) {
                expInd += spatialCoord[j]*sharedCoords[i](j);            
            }
            
            // Range reduce exponential index. 
            expInd = expInd&(gridDims[0]-1);// Optimisation for gridDims[0] a power of two
            
            // Get complex exponential
            T expArg = expInd*expArgFactor;
            T expVal[2];
            sincosIntrinsic(expArg, &(expVal[1]), &(expVal[0]));
            
            // Add real part of term
            val = fmaIntrinsic(sharedCoeffs[i].re, expVal[0], val);
            val = fmaIntrinsic(sharedCoeffs[i].im, expVal[1], val);
        }

        // Sync after compute
        __syncthreads();
    }
    
    // Return
    return val;
}

// Kernel
template <typename T, unsigned int N>
__global__ void GET_IDFT_REAL(SpectralDatabaseCUDA<T,N> *db, unsigned int dsetIdx, unsigned int *spatialCoord, T *val) {
    *val = db->getIDFTRealD(dsetIdx, spatialCoord);
}

///////////////////////////////
// SPECTRAL DATABASE UNIFIED //
///////////////////////////////

/**
 * @class SpectralDataUnifiedCUDA
 * @author Michael Malahe
 * @date 27/04/17
 * @file spectralUtilsCUDA.h
 * @brief 
 * @tparam T the scalar type
 * @tparam N the dimension of the data
 * @tparam P the number of coefficients tied to each coordinate
 */
template<typename T, unsigned int N, unsigned int P>
struct ALIGN(16) SpectralDataUnifiedCUDA {
    SpectralCoordCUDA<N> coord;
    SpectralCoeffCUDA<T> coeffs[P];
};

/**
 * @class SpectralDatabaseUnifiedCUDA
 * @author Michael Malahe
 * @date 26/04/17
 * @file spectralUtilsCUDA.h
 * @brief A spectral database for DFTs of real data.
 * @details This is specifically for "unified" coefficient datasets, which
 * here means that there is a single ordering of the dominant components that
 * all of the coefficients follow. See hpp::SpectralDatabaseCUDA
 * @tparam T the scalar datatype
 * @tparam N the dimension of the database
 * @tparam P the number of coefficients tied to each coordinate
 */
template<typename T, unsigned int N, unsigned int P>
class SpectralDatabaseUnifiedCUDA
{
public:
    SpectralDatabaseUnifiedCUDA();
    SpectralDatabaseUnifiedCUDA(const SpectralDatabaseUnified<T>& dbIn, const std::vector<SpectralDatasetID>& dsetIDs);
    __device__ void getIDFTRealDShared(unsigned int *spatialCoord, T *outputs, unsigned int nShared, SpectralDataUnifiedCUDA<T,N,P> *sharedData) const;
    __device__ void getIDFTRealDSharedPair(unsigned int *spatialCoord0, T *outputs0, unsigned int *spatialCoord1, T *outputs1, unsigned int nShared, SpectralDataUnifiedCUDA<T,N,P> *sharedData) const;
    // Getters
    __device__ T* getGridStarts() {return gridStarts;}
    __device__ T* getGridSteps() {return gridSteps;}
    __device__ unsigned int * getGridDims() {return gridDims;}   
    unsigned int getNDsets() const {return nDsets;}
    unsigned int getNTerms() const {return nTerms;}
    
protected:
    
private:    
    // Grid dimensions    
    unsigned int *gridDims;
    std::shared_ptr<unsigned int> gridDimsSharedPtr;
    
    // Grid spatial parameters    
    T *gridStarts;
    std::shared_ptr<T> gridStartsSharedPtr;    
    T *gridSteps;
    std::shared_ptr<T> gridStepsSharedPtr;
    
    // Number of datasets
    unsigned int nDsets;
    
    // Number of terms
    unsigned int nTerms;

    // Spectral data
    SpectralDataUnifiedCUDA<T,N,P> *data;
    
    // Shared pointers to assure correct copying and destruction
    std::shared_ptr<SpectralDataUnifiedCUDA<T,N,P>> dataSharedPtr;
};

template <typename T, unsigned int N>
__device__ void getExpVal(unsigned int *spatialCoord, SpectralCoordCUDA<N>& coord, unsigned int gridDim, T expArgFactor, T* expValRe, T* expValIm) {
    // Get exponential index
    unsigned int expInd = spatialCoord[0]*coord(0);
    for (unsigned int j=1; j<N; j++) {
        expInd += spatialCoord[j]*coord(j);            
    }
    
    // Range reduce exponential index. This is a significant saving 
    // compared with having the intrinsic trig functions do it. 
    expInd = expInd&(gridDim-1);// Optimisation for gridDim a power of two
    
    // Get complex exponential
    T expArg = expInd*expArgFactor;
    sincosIntrinsic(expArg, expValIm, expValRe);
}

/**
 * @brief Device IDFTD
 * @details See hpp::SpectralDatabaseCUDA::getIDFTRealDShared
 * @param dsetIdx
 * @param spatialCoord
 * @param nShared
 * @param sharedCoords
 * @param sharedCoeffs
 */
template <typename T, unsigned int N, unsigned int P>
__device__ void SpectralDatabaseUnifiedCUDA<T,N,P>::getIDFTRealDShared(unsigned int *spatialCoord, T *outputs, unsigned int nShared, SpectralDataUnifiedCUDA<T,N,P> *sharedData) const {    
    // Commonly used global memory into registers
    const unsigned int gridDimReg = gridDims[0];
    const unsigned int nTermsReg = nTerms;
    
    // Exponential argument common factor
    T expArgFactor = 2*((T)M_PI)/gridDimReg;
    
    // The data is interpreted as ints for the purposes of coalesced
    // reading from global memory to shared memory
    int *globalDataAsInt = (int*)data;
    int *sharedDataAsInt = (int*)sharedData;
    unsigned int readElementSize = sizeof(int);
    
    // Read into shared memory as a block
    // termsPerRead is the number of shared values that are read per large read
    // nReads is the number of times that we must perform a large read from
    // global memory to shared memory.
    unsigned int termsPerLargeRead = nShared;
    unsigned int elementsPerLargeRead = termsPerLargeRead*sizeof(SpectralDataUnifiedCUDA<T,N,P>)/readElementSize;
    unsigned int totalElementsToRead = nTermsReg*sizeof(SpectralDataUnifiedCUDA<T,N,P>)/readElementSize;
    unsigned int nReads = nTermsReg/termsPerLargeRead;
    if (nTermsReg % termsPerLargeRead != 0) nReads++;
    
    // elementsPerBlockRead is the number of values that are read when every thread 
    // in the block does a single read
    unsigned int elementsPerBlockRead = blockDim.x;
    
    // Initial values
    for (unsigned int i=0; i<P; i++) {
        outputs[i] = (T)0.0;
    }
    
    // Loop by shared memory reads
    for (unsigned int iRead=0; iRead<nReads; iRead++) {
        // Read data into shared memory
        unsigned int dataReadStartGlobal = iRead*elementsPerLargeRead+threadIdx.x;
        unsigned int dataReadEndGlobal = umin((iRead+1)*elementsPerLargeRead, totalElementsToRead);
        for (unsigned int readIdxGlobal = dataReadStartGlobal; readIdxGlobal<dataReadEndGlobal; readIdxGlobal+=elementsPerBlockRead) {
            unsigned int readIdxShared = readIdxGlobal%elementsPerLargeRead;
            sharedDataAsInt[readIdxShared] = globalDataAsInt[readIdxGlobal];
        }
        
        // Sync after read
        __syncthreads();
        
        // Range of terms to compute
        unsigned int termsStartGlobal = iRead*termsPerLargeRead;
        unsigned int termsEndGlobal = umin(termsStartGlobal+termsPerLargeRead, nTermsReg);
        unsigned int termsStartShared = 0;
        unsigned int termsEndShared = termsEndGlobal-termsStartGlobal;
        
        // Compute terms
        for (unsigned int i=termsStartShared; i<termsEndShared; i++) {
            // Get exponential index
            unsigned int expInd = spatialCoord[0]*sharedData[i].coord(0);
            for (unsigned int j=1; j<N; j++) {
                expInd += spatialCoord[j]*sharedData[i].coord(j);            
            }
            
            // Range reduce exponential index. This is a significant saving 
            // compared with having the intrinsic trig functions do it. 
            expInd = expInd&(gridDimReg-1);// Optimisation for gridDims[0] a power of two
            
            // Get complex exponential
            T expArg = expInd*expArgFactor;
            T expVal[2];
            sincosIntrinsic(expArg, &(expVal[1]), &(expVal[0]));
            
            // Update values
            for (unsigned int iDset = 0; iDset<P; iDset++) {
                outputs[iDset] = fmaIntrinsic(sharedData[i].coeffs[iDset].re, expVal[0], outputs[iDset]);
                outputs[iDset] = fmaIntrinsic(sharedData[i].coeffs[iDset].im, expVal[1], outputs[iDset]);
            }
        }

        // Sync after compute
        __syncthreads();
    }
}

/**
 * @brief Device IDFTD
 * @details See hpp::SpectralDatabaseCUDA::getIDFTRealDShared. This version
 * works on two sets of spatial coordinates at once, to reduce pressure
 * on the shared memory system
 * @param dsetIdx
 * @param spatialCoord
 * @param nShared
 * @param sharedCoords
 * @param sharedCoeffs
 */
template <typename T, unsigned int N, unsigned int P>
__device__ void SpectralDatabaseUnifiedCUDA<T,N,P>::getIDFTRealDSharedPair(unsigned int *spatialCoord0, T *outputs0, unsigned int *spatialCoord1, T *outputs1, unsigned int nShared, SpectralDataUnifiedCUDA<T,N,P> *sharedData) const {    
    // Commonly used global memory into registers
    const unsigned int gridDimReg = gridDims[0];
    const unsigned int nTermsReg = nTerms;
    
    // Exponential argument common factor
    T expArgFactor = 2*((T)M_PI)/gridDimReg;
    
    // The data is interpreted as ints for the purposes of coalesced
    // reading from global memory to shared memory
    int *globalDataAsInt = (int*)data;
    int *sharedDataAsInt = (int*)sharedData;
    unsigned int readElementSize = sizeof(int);
    
    // Read into shared memory as a block
    // termsPerRead is the number of shared values that are read per large read
    // nReads is the number of times that we must perform a large read from
    // global memory to shared memory.
    unsigned int termsPerLargeRead = nShared;
    unsigned int elementsPerLargeRead = termsPerLargeRead*sizeof(SpectralDataUnifiedCUDA<T,N,P>)/readElementSize;
    unsigned int totalElementsToRead = nTermsReg*sizeof(SpectralDataUnifiedCUDA<T,N,P>)/readElementSize;
    unsigned int nReads = nTermsReg/termsPerLargeRead;
    if (nTermsReg % termsPerLargeRead != 0) nReads++;
    
    // elementsPerBlockRead is the number of values that are read when every thread 
    // in the block does a single read
    unsigned int elementsPerBlockRead = blockDim.x;
    
    // Initial values
    for (unsigned int i=0; i<P; i++) {
        outputs0[i] = (T)0.0;
        outputs1[i] = (T)0.0;
    }
    
    // Loop by shared memory reads
    for (unsigned int iRead=0; iRead<nReads; iRead++) {
        // Read data into shared memory
        unsigned int dataReadStartGlobal = iRead*elementsPerLargeRead+threadIdx.x;
        unsigned int dataReadEndGlobal = umin((iRead+1)*elementsPerLargeRead, totalElementsToRead);
        for (unsigned int readIdxGlobal = dataReadStartGlobal; readIdxGlobal<dataReadEndGlobal; readIdxGlobal+=elementsPerBlockRead) {
            unsigned int readIdxShared = readIdxGlobal%elementsPerLargeRead;
            sharedDataAsInt[readIdxShared] = globalDataAsInt[readIdxGlobal];
        }
        
        // Sync after read
        __syncthreads();
        
        // Range of terms to compute
        unsigned int termsStartGlobal = iRead*termsPerLargeRead;
        unsigned int termsEndGlobal = umin(termsStartGlobal+termsPerLargeRead, nTermsReg);
        unsigned int termsStartShared = 0;
        unsigned int termsEndShared = termsEndGlobal-termsStartGlobal;
        
        // Compute terms
        for (unsigned int i=termsStartShared; i<termsEndShared; i++) {
            // Read in coordinate and coefficient
            SpectralDataUnifiedCUDA<T,N,P> unifiedData = sharedData[i];
            
            // FIRST SPATIAL COORDINATE
            T expValRe, expValIm;
            getExpVal(spatialCoord0, unifiedData.coord, gridDimReg, expArgFactor, &expValRe, &expValIm);
            for (unsigned int iDset = 0; iDset<P; iDset++) {
                outputs0[iDset] = fmaIntrinsic(unifiedData.coeffs[iDset].re, expValRe, outputs0[iDset]);
                outputs0[iDset] = fmaIntrinsic(unifiedData.coeffs[iDset].im, expValIm, outputs0[iDset]);
            }
            
            // SECOND SPATIAL COORDINATE
            getExpVal(spatialCoord1, unifiedData.coord, gridDimReg, expArgFactor, &expValRe, &expValIm);
            for (unsigned int iDset = 0; iDset<P; iDset++) {
                outputs1[iDset] = fmaIntrinsic(unifiedData.coeffs[iDset].re, expValRe, outputs1[iDset]);
                outputs1[iDset] = fmaIntrinsic(unifiedData.coeffs[iDset].im, expValIm, outputs1[iDset]);
            }            
        }

        // Sync after compute
        __syncthreads();
    }
}

#endif /* HPP_USE_CUDA */
}//END NAMESPACE HPP

#endif /* HPP_SPECTRAL_UTILS_CUDA_H */