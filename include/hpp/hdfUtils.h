/**
* @file hdfUtils.h
* @author Michael Malahe
* @brief Header file for helper functions with HDF.
* @details Uses the C interface and parallel implementation.
*/

#ifndef HDFUTILS_H
#define HDFUTILS_H

#include <stdexcept>
#include <cstddef>
#include <string>
#include <stdio.h>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include "mpi.h"
#include <hpp/config.h>
#include <hdf5/openmpi/hdf5.h>

namespace hpp
{

class HDFUtilsError: public std::runtime_error
{
    public:
        HDFUtilsError (const std::string &val) : std::runtime_error::runtime_error(val) {}
};    

// Check an HDF C API call
#define HDFCHECK(ans) hdfAssert((ans), __FILE__, __LINE__)
inline void hdfAssert(herr_t code, const char *file, int line){
    if (code < 0){        
        // Throw
        std::ostringstream errorStringStream;
        errorStringStream << "HDF error above occured at "; 
        errorStringStream << file << ":" << line;
        throw HDFUtilsError(errorStringStream.str());
    }
}

// Suppressing errors
extern H5E_auto2_t dummyHDFErrorHandler;
extern void *dummyHDFClientData;
void hdfSuppressErrors();
void hdfRestoreErrors();
#define HDF_IGNORE_ERRORS(expr) hdfSuppressErrors();expr;hdfRestoreErrors()

enum class HDFReadWrite: char
{ 
    Write='w',
    Read='r'
};

/**
 * @brief A complex number type.
 * @details Designed to conform to the H5py implementation of complex numbers
 * 
*/
struct hdf_complex_t{
    double r;
    double i;
};
extern hid_t hdf_complex_id;

/**
 * @brief Get HDF5 equivalent type
 * @details For the C interface
 * @return The HDF5 type
 */
template <typename T>
hid_t getHDF5TypeC() {
    hid_t dataType;
    if (std::is_same<T, float>::value) {
        dataType = H5T_NATIVE_FLOAT;
    }
    else if (std::is_same<T, double>::value) {
        dataType = H5T_NATIVE_DOUBLE;
    }
    else if (std::is_same<T, unsigned short>::value) {
        dataType = H5T_NATIVE_USHORT;
    }
    else if (std::is_same<T, unsigned int>::value) {
        dataType = H5T_NATIVE_UINT;
    }
    else if (std::is_same<T, hdf_complex_t>::value) {
        dataType = hdf_complex_id;
    }
    else {
        throw HDFUtilsError("Datatype lookup not implemented for this type.");
    }
    return dataType;
};

struct HDFReadWriteParamsC {
    std::vector<hsize_t> dataOffset;
    std::vector<hsize_t> dataCount;
    hid_t datatype;
};

template <typename T>
hid_t createHDF5Dataset(hid_t file_id, std::string datasetName, std::vector<hsize_t> dataDims) {
    // Create the dataset
    hsize_t dataRank = dataDims.size();
    
    // Dataspace
    hid_t dataspace = H5Screate_simple(dataRank, dataDims.data(), NULL);
    HDFCHECK(dataspace);
    
    // Datatype
    hid_t dataType = getHDF5TypeC<T>();
    
    // Actual dataset 
    hid_t dataset = H5Dcreate(file_id, datasetName.c_str(), dataType, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    HDFCHECK(dataset);  
    
    // Close
    //HDFCHECK(H5Sclose(dataspace)); // this needs to be closed after the dataset is
    
    // Return
    return dataset;
}

template <typename T>
hid_t createHDF5GridOfArrays(hid_t file_id, std::string datasetName, std::vector<hsize_t> gridDims, std::vector<hsize_t> arrayDims) {
    // Get full data dims
    std::vector<hsize_t> dataDims = gridDims;
    dataDims.insert(dataDims.end(), arrayDims.begin(), arrayDims.end());
    
    // Create
    return createHDF5Dataset<T>(file_id, datasetName, dataDims);
}

/**
 * @brief Gets the parameters and ensures that they're compatible with the actual dataset.
 * @param dset_id
 * @param plist_id
 * @param gridOffset
 * @param arrayDims
 * @param output
 */
template <typename T>
HDFReadWriteParamsC getReadWriteParametersForMultipleHDF5Arrays(hid_t dset_id, std::vector<hsize_t> gridOffset, 
                                              std::vector<hsize_t> arrayDims, std::vector<hsize_t> arrayCount) 
{
    // Grid: the grid of arrays, dimensions (m_1, m_2, ... )
    // Array: the array, dimensions (n_1, n_2, ... )
    // Data: The combination, dimensions (m1, m2, ..., n1, n_2, ...)
    // Offset: The offset of the array within the grid
    
    // Check that datatypes match
    hid_t datatype = H5Dget_type(dset_id);
    HDFCHECK(datatype);
    hid_t datatypeExpected = getHDF5TypeC<T>();
    if (!H5Tequal(datatype,datatypeExpected)) {
        throw HDFUtilsError("Datatype mismatch.");
    }
    HDFCHECK(H5Tclose(datatype));
    datatype = datatypeExpected;
    
    // Get data dimensions
    hid_t dataspace = H5Dget_space(dset_id);
    HDFCHECK(dataspace);
    hsize_t dataRank = H5Sget_simple_extent_ndims(dataspace);
    HDFCHECK(dataRank);
    std::vector<hsize_t> dataDims(dataRank);
    H5Sget_simple_extent_dims(dataspace, dataDims.data(), NULL);
    
    // Check rank matches
    hsize_t gridRank = gridOffset.size();
    hsize_t arrayRank = arrayDims.size();
    hsize_t dataRankExpected = gridRank + arrayRank;
    if (dataRank != dataRankExpected) {
        throw HDFUtilsError("Data rank mismatch.");
    }
    
    // Check array dimensions match
    std::vector<hsize_t> arrayDimsRead(dataDims.end()-arrayDims.size(), dataDims.end());
    if (arrayDims != arrayDimsRead) {
        throw HDFUtilsError("Array dimension mismatch.");
    }
    
    // Get the grid dimensions
    std::vector<hsize_t> gridDims(dataDims.begin(), dataDims.begin()+gridRank);
    
    // Check that offset is within the grid bounds
    for (unsigned int i=0; i<gridRank; i++) {
        if (gridOffset[i] > gridDims[i]-1) {
            throw HDFUtilsError("Offset out of grid bounds.");
        }
    }
    
    // The offset and count in the overall data
    std::vector<hsize_t> dataOffset(dataRank);
    std::vector<hsize_t> dataCount(dataRank);
    for (unsigned int i=0; i<gridRank; i++) {
        dataOffset[i] = gridOffset[i];
        dataCount[i] = arrayCount[i];
    }
    for (unsigned int i=0; i<arrayRank; i++) {
        dataOffset[gridRank + i] = 0;
        dataCount[gridRank + i] = arrayDims[i];
    }    
    
    // Close any spaces created inside this function
    HDFCHECK(H5Sclose(dataspace));
    
    // Prepare parameters
    HDFReadWriteParamsC parms;
    parms.dataOffset = dataOffset;
    parms.dataCount = dataCount;
    parms.datatype = datatype;
    
    // Return
    return parms;  
}

/**
 * @brief Gets the parameters and ensures that they're compatible with the actual dataset.
 * @param dset_id
 * @param plist_id
 * @param gridOffset
 * @param arrayDims
 * @param output
 */
template <typename T>
HDFReadWriteParamsC getReadWriteParametersForSingleHDF5Array(hid_t dset_id, std::vector<hsize_t> gridOffset, 
                                              std::vector<hsize_t> arrayDims) 
{    
    unsigned int gridRank = gridOffset.size();
    std::vector<hsize_t> arrayCount(gridRank, 1);
    return getReadWriteParametersForMultipleHDF5Arrays<T>(dset_id, gridOffset, arrayDims, arrayCount);
}

template <typename T>
void readWriteHDF5SimpleArray(hid_t dset_id, hid_t plist_id, HDFReadWriteParamsC parms, T* output, HDFReadWrite mode) {    
    // Create the spaces
    hid_t dataspace = H5Dget_space(dset_id);
    HDFCHECK(dataspace);
    hid_t memspace = H5Screate_simple(parms.dataCount.size(), parms.dataCount.data(), NULL); 
    HDFCHECK(memspace);
    
    // Select the hyperslab
    HDFCHECK(H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, parms.dataOffset.data(), NULL, parms.dataCount.data(), NULL));
    
    // Read/write
    if (mode == HDFReadWrite::Read) {
        HDFCHECK(H5Dread(dset_id, parms.datatype, memspace, dataspace, plist_id, output));
    }
    else if (mode == HDFReadWrite::Write) {
        HDFCHECK(H5Dwrite(dset_id, parms.datatype, memspace, dataspace, plist_id, output));
    }
    else {
        throw HDFUtilsError("Unrecognized mode.");
    }
    
    // Close any spaces created inside this function
    HDFCHECK(H5Sclose(memspace));
    HDFCHECK(H5Sclose(dataspace));
}

template <typename T>
void readHDF5SimpleArray(hid_t dset_id, hid_t plist_id, HDFReadWriteParamsC parms, T* output) {    
    readWriteHDF5SimpleArray<T>(dset_id, plist_id, parms, output, HDFReadWrite::Read);
}

template <typename T>
void writeHDF5SimpleArray(hid_t dset_id, hid_t plist_id, HDFReadWriteParamsC parms, T* output) {    
    readWriteHDF5SimpleArray<T>(dset_id, plist_id, parms, output, HDFReadWrite::Write);
}

template <typename T>
void readSingleHDF5Array(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> gridOffset, std::vector<hsize_t> arrayDims, T* output) {    
    HDFReadWriteParamsC parms = getReadWriteParametersForSingleHDF5Array<T>(dset_id, gridOffset, arrayDims);
    readHDF5SimpleArray<T>(dset_id, plist_id, parms, output);
}

template <typename T>
void readSingleHDF5Array(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> arrayDims, T* output) {    
    std::vector<hsize_t> gridOffset;//no offset
    HDFReadWriteParamsC parms = getReadWriteParametersForSingleHDF5Array<T>(dset_id, gridOffset, arrayDims);
    readHDF5SimpleArray<T>(dset_id, plist_id, parms, output);
}

template <typename T>
void writeSingleHDF5Array(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> gridOffset, std::vector<hsize_t> arrayDims, T* output) {    
    HDFReadWriteParamsC parms = getReadWriteParametersForSingleHDF5Array<T>(dset_id, gridOffset, arrayDims);
    writeHDF5SimpleArray<T>(dset_id, plist_id, parms, output);
}

template <typename T>
void writeSingleHDF5Array(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> arrayDims, T* output) {    
    std::vector<hsize_t> gridOffset;//no offset
    HDFReadWriteParamsC parms = getReadWriteParametersForSingleHDF5Array<T>(dset_id, gridOffset, arrayDims);
    writeHDF5SimpleArray<T>(dset_id, plist_id, parms, output);
}

template <typename T>
void writeMultipleHDF5Arrays(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> gridOffset, std::vector<hsize_t> arrayDims, std::vector<hsize_t> arrayCount, T* output) {    
    HDFReadWriteParamsC parms = getReadWriteParametersForMultipleHDF5Arrays<T>(dset_id, gridOffset, arrayDims, arrayCount);
    writeHDF5SimpleArray<T>(dset_id, plist_id, parms, output);
}

template <typename T>
void readMultipleHDF5Arrays(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> gridOffset, std::vector<hsize_t> arrayDims, std::vector<hsize_t> arrayCount, T* output) {    
    HDFReadWriteParamsC parms = getReadWriteParametersForMultipleHDF5Arrays<T>(dset_id, gridOffset, arrayDims, arrayCount);
    readHDF5SimpleArray<T>(dset_id, plist_id, parms, output);
}

template <typename T>
void readSingleHDF5Value(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> gridOffset, T* output) {
    // Empty array dims is equivalent to a single value
    std::vector<hsize_t> arrayDims;
    HDFReadWriteParamsC parms = getReadWriteParametersForSingleHDF5Array<T>(dset_id, gridOffset, arrayDims);
    readHDF5SimpleArray<T>(dset_id, plist_id, parms, output);
}

template <typename T>
void writeSingleHDF5Value(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> gridOffset, T* output) {    
    // Empty array dims is equivalent to a single value
    std::vector<hsize_t> arrayDims;
    HDFReadWriteParamsC parms = getReadWriteParametersForSingleHDF5Array<T>(dset_id, gridOffset, arrayDims);
    writeHDF5SimpleArray<T>(dset_id, plist_id, parms, output);
}

std::vector<hsize_t> getDatasetDims(hid_t dset_id);

// HANDLER FOR MPI I/O //
/////////////////////////

/**
 * @class HDF5MPIHandler
 * @author Michael
 * @date 07/12/16
 * @file hdfUtils.h
 * @brief 
 * @details
 * @todo Currently it's unclear how to correctly open and read datasets in parallel
 * using the HDF5 C++ API, so the only functionality here is creating, writing to and
 * closing datasets.
 */
class HDF5MPIHandler
{
    public:
        HDF5MPIHandler(std::string filename, MPI_Comm comm, bool doCreate);
        
        // Create datasets
        template <typename T>
        hid_t createDataset(std::string datasetName, std::vector<hsize_t> dataDims);     
        template <typename T>
        hid_t createDataset(std::string datasetName, std::vector<hsize_t> gridDims, std::vector<hsize_t> arrayDims);        
        
        // Get dataset
        hid_t getDataset(std::string datasetName);
        
        // Property list for transfer
        hid_t getPropertyListTransferIndependent() {return plist_id_xfer_independent;};
        
        // Get names of datasets
        std::vector<std::string> getDatasetNames();
        
        // Default destructor
        ~HDF5MPIHandler();
    private:
        // File
        std::string filename;
        hid_t file_id;
        
        // MPI Configuration
        MPI_Comm comm;
        int comm_size;
        int comm_rank;
        
        // Dictionary of datasets
        // Open datasets get added to or removed from this dictionary
        // All the ones left open at the time of destruction are closed
        std::map<std::string, hid_t> currentlyOpenDatasets;
        
        // Datasets
        void openDataset(std::string datasetName);
        void closeDataset(std::string datasetName);
        
        // Property lists
        hid_t plist_id_file_access;
        hid_t plist_id_xfer_independent;
};

} //END NAMESPACE HPP

#endif /* HDFUTILS_H */