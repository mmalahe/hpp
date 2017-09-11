/** @file hdfUtilsCpp.h
* @author Michael Malahe
* @brief Header file for helper functions with HDF, C++ API
*/

#ifndef HDFUTILSCPP_H
#define HDFUTILSCPP_H

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
#include <hdf5/serial/H5Cpp.h>
#include <hpp/hdfUtils.h>

namespace hpp
{

/**
 * @brief Get HDF5 equivalent type
 * @details For C++ interface
 * @return The HDF5 type
 */
template <typename T>
H5::DataType getHDF5Type() {
    H5::DataType dataType;
    if (std::is_same<T, float>::value) {
        dataType = H5::PredType::NATIVE_FLOAT;
    }
    else if (std::is_same<T, double>::value) {
        dataType = H5::PredType::NATIVE_DOUBLE;
    }
    else {
        throw HDFUtilsError("Datatype lookup not implemented for this type.");
    }
    return dataType;
}

struct HDFReadWriteParams {
    std::vector<hsize_t> dataOffset;
    std::vector<hsize_t> dataCount;
    H5::DataType datatype;
};

template <typename T>
H5::DataSet createHDF5Dataset(H5::H5File& file, const H5std_string& datasetName, std::vector<hsize_t> dataDims) {
    // Create the dataset
    hsize_t dataRank = dataDims.size();
    H5::DataSpace dataspace;
    if (dataRank == 0) {
        dataspace = H5::DataSpace();
    }
    else {
        dataspace = H5::DataSpace(dataRank, dataDims.data());
    }
    H5::DataType dataType = getHDF5Type<T>();
    H5::DataSet dataset = file.createDataSet(datasetName.c_str(), dataType, dataspace);
    
    // Return
    return dataset;
}

template <typename T>
H5::DataSet createHDF5DatasetScalar(H5::H5File& file, const H5std_string& datasetName) {
    std::vector<hsize_t> dataDims;
    return createHDF5Dataset<T>(file, datasetName, dataDims);
}

template <typename T>
H5::DataSet createHDF5GridOfArrays(H5::H5File& file, const H5std_string& datasetName, std::vector<hsize_t> gridDims, std::vector<hsize_t> arrayDims) {
    // Get full data dims
    std::vector<hsize_t> dataDims = gridDims;
    dataDims.insert(dataDims.end(), arrayDims.begin(), arrayDims.end());
    
    // Create
    return createHDF5Dataset<T>(file, datasetName, dataDims);
}

template <typename T>
HDFReadWriteParams getReadWriteParametersForSingleHDF5Array(H5::DataSet& dataset, std::vector<hsize_t> gridOffset, 
                                                             std::vector<hsize_t> arrayDims) 
{
    // Grid: the grid of arrays, dimensions (m_1, m_2, ... )
    // Array: the array, dimensions (n_1, n_2, ... )
    // Data: The combination, dimensions (m1, m2, ..., n1, n_2, ...)
    // Offset: The offset of the array within the grid
    
    // Check that datatypes match
    H5::DataType datatype = dataset.getDataType();
    H5::DataType datatypeExpected = getHDF5Type<T>();
    if (!(datatype == datatypeExpected)) { //no "!=" in API
        throw HDFUtilsError("Datatype mismatch.");
    }
    
    // Get data dimensions
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dataRank = dataspace.getSimpleExtentNdims();
    std::vector<hsize_t> dataDims(dataRank);
    dataspace.getSimpleExtentDims(dataDims.data());
    
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
        dataCount[i] = 1;
    }
    for (unsigned int i=0; i<arrayRank; i++) {
        dataOffset[gridRank + i] = 0;
        dataCount[gridRank + i] = arrayDims[i];
    }
    
    // Prepare parameters
    HDFReadWriteParams parms;
    parms.dataOffset = dataOffset;
    parms.dataCount = dataCount;
    parms.datatype = datatype;
    
    // Return
    return parms;  
}

template <typename T>
void readWriteHDF5SimpleArray(H5::DataSet& dataset, HDFReadWriteParams parms, T* output, HDFReadWrite mode)
{
    // Dataspace
    H5::DataSpace dataspace = dataset.getSpace();
    
    // Create the memspace for the read
    H5::DataSpace memspace(parms.dataCount.size(), parms.dataCount.data());
    
    // Select the hyperslab
    dataspace.selectHyperslab(H5S_SELECT_SET, parms.dataCount.data(), parms.dataOffset.data());
    
    // Read/write
    if (mode == HDFReadWrite::Read) {
        dataset.read(output, parms.datatype, memspace, dataspace);
    }
    else if (mode == HDFReadWrite::Write) {
        dataset.write(output, parms.datatype, memspace, dataspace);
    }
    else {
        throw HDFUtilsError("Unrecognized mode.");
    }    
}

template <typename T>
void readHDF5SimpleArray(H5::DataSet& dataset, HDFReadWriteParams parms, T* output) {    
    readWriteHDF5SimpleArray<T>(dataset, parms, output, HDFReadWrite::Read);
}

template <typename T>
void writeHDF5SimpleArray(H5::DataSet& dataset, HDFReadWriteParams parms, T* output) {    
    readWriteHDF5SimpleArray<T>(dataset, parms, output, HDFReadWrite::Write);
}

template <typename T>
void readSingleHDF5Array(H5::DataSet& dataset, std::vector<hsize_t> gridOffset, std::vector<hsize_t> arrayDims, T* output) {    
    HDFReadWriteParams parms = getReadWriteParametersForSingleHDF5Array<T>(dataset, gridOffset, arrayDims);
    readHDF5SimpleArray<T>(dataset, parms, output);
}

template <typename T>
void writeSingleHDF5Array(H5::DataSet& dataset, std::vector<hsize_t> gridOffset, std::vector<hsize_t> arrayDims, T* output) {    
    HDFReadWriteParams parms = getReadWriteParametersForSingleHDF5Array<T>(dataset, gridOffset, arrayDims);
    writeHDF5SimpleArray<T>(dataset, parms, output);
}

template <typename T>
void writeVectorToHDF5Array(H5::H5File& file, const std::string& dsetName, std::vector<T>& vec) {
    std::vector<hsize_t> dataOffset;
    std::vector<hsize_t> dataDims = {vec.size()};    
    H5::DataSet dset = createHDF5Dataset<T>(file, dsetName, dataDims);
    writeSingleHDF5Array<T>(dset, dataOffset, dataDims, vec.data());
}

} //END NAMESPACE HPP

#endif /* HDFUTILS_CPP */