/** 
* @file hdfUtils.cpp
* @author Michael Malahe
* @brief Implementation for functions in hdfUtils.h
* @details Note that these functions can only deal with the C interface.
*/

#include <hpp/hdfUtils.h>

namespace hpp
{

// Suppressing errors
H5E_auto2_t dummyHDFErrorHandler;
void *dummyHDFClientData;
void hdfSuppressErrors() {
     H5Eget_auto(H5E_DEFAULT, &dummyHDFErrorHandler, &dummyHDFClientData);
     H5Eset_auto(H5E_DEFAULT, NULL, NULL);
}
void hdfRestoreErrors() {
    H5Eset_auto(H5E_DEFAULT, dummyHDFErrorHandler, dummyHDFClientData);
}

// Complex type
hid_t getComplexHDFType() {
    hid_t complex_type = H5Tcreate(H5T_COMPOUND, sizeof(hdf_complex_t));
    HDFCHECK(complex_type);
    HDFCHECK(H5Tinsert(complex_type, "r", 0, H5T_NATIVE_DOUBLE));
    HDFCHECK(H5Tinsert(complex_type, "i", sizeof(double), H5T_NATIVE_DOUBLE));
    return complex_type;
}
hid_t hdf_complex_id = getComplexHDFType();

// Dataset dimensions
std::vector<hsize_t> getDatasetDims(hid_t dset_id) {
    hid_t dspace_id = H5Dget_space(dset_id);
    int ndims = H5Sget_simple_extent_ndims(dspace_id);
    HDFCHECK(ndims);
    std::vector<hsize_t> dims(ndims);
    HDFCHECK(H5Sget_simple_extent_dims(dspace_id, dims.data(), NULL));
    HDFCHECK(H5Sclose(dspace_id));
    return dims;
}

/**
 * @brief Create a handler for HDF5 parallel I/O on a file
 * @param filename the name of the file
 * @param comm the MPI communicator
 * @param doCreate if true, overwrite the file if it exists
 * @return 
 */
HDF5Handler::HDF5Handler(std::string filename, MPI_Comm comm, bool doCreate) {
    usingMPI = true;
    
    MPI_Barrier(comm);
    
    // Basics
    this->filename = filename;
    this->comm = comm;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Properties for file for access
    // plist_id_file_access is a "file access" property list    
    this->plist_id_file_access = H5Pcreate(H5P_FILE_ACCESS);
    HDFCHECK(this->plist_id_file_access);
    
    // Note that this is operation is not checked because the HDF5 API claims that
    // plist_id_file_access is not a "file access" property list, and raises
    // an error.
    HDF_IGNORE_ERRORS(H5Pset_fapl_mpio(this->plist_id_file_access, comm, MPI_INFO_NULL));
    
    // Create file if requested
    if (doCreate) {
        hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, this->plist_id_file_access);
        HDFCHECK(H5Fclose(file));
    }    

    // Open file collectively with parallel I/O access    
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, this->plist_id_file_access);
    HDFCHECK(file_id);
    
    // Properties for independent I/O
    // plist_id_xfer_independent is a "data transfer" Property List
    this->plist_id_xfer_independent = H5Pcreate(H5P_DATASET_XFER);
    HDFCHECK(this->plist_id_xfer_independent);
    
    // Note that this is operation is not checked because the HDF5 API claims that
    // plist_id_xfer_independent is not a "data transfer" property list, and raises
    // an error.
    HDF_IGNORE_ERRORS(H5Pset_dxpl_mpio(this->plist_id_xfer_independent, H5FD_MPIO_INDEPENDENT));
    
    MPI_Barrier(comm);
}

/**
 * @brief Create a handler for HDF5 serial I/O on a file
 * @param filename filename the name of the file
 * @param doCreate if true, overwrite the file if it exists
 * @return 
 */
HDF5Handler::HDF5Handler(std::string filename, bool doCreate) {    
    usingMPI = false;
    
    // Basics
    this->filename = filename;
    
    // Properties for file for access
    // plist_id_file_access is a "file access" property list    
    this->plist_id_file_access = H5Pcreate(H5P_FILE_ACCESS);
    HDFCHECK(this->plist_id_file_access);
    
    // Create file if requested
    if (doCreate) {
        hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, this->plist_id_file_access);
        HDFCHECK(H5Fclose(file));
    }    

    // Open file   
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, this->plist_id_file_access);
    HDFCHECK(file_id);
    
    // Properties for independent I/O
    // plist_id_xfer_independent is a "data transfer" Property List
    this->plist_id_xfer_independent = H5Pcreate(H5P_DATASET_XFER);
    HDFCHECK(this->plist_id_xfer_independent);
}

hid_t HDF5Handler::getDataset(std::string datasetName) {    
    if (currentlyOpenDatasets.count(datasetName) == 0) {
        this->openDataset(datasetName);
    }
    return currentlyOpenDatasets[datasetName];
}

template <typename T>
hid_t HDF5Handler::createDataset(std::string datasetName, std::vector<hsize_t> dataDims) {                
    if (usingMPI) {MPI_Barrier(comm);}
    hid_t dset_id = createHDF5Dataset<T>(file_id, datasetName, dataDims);
    currentlyOpenDatasets.insert(std::pair<std::string, hid_t>(datasetName, dset_id));
    if (usingMPI) {MPI_Barrier(comm);}
    return dset_id;
}
template hid_t HDF5Handler::createDataset<float>(std::string datasetName, std::vector<hsize_t> dataDims);
template hid_t HDF5Handler::createDataset<double>(std::string datasetName, std::vector<hsize_t> dataDims);
template hid_t HDF5Handler::createDataset<hdf_complex_t>(std::string datasetName, std::vector<hsize_t> dataDims);
template hid_t HDF5Handler::createDataset<unsigned short>(std::string datasetName, std::vector<hsize_t> dataDims);

template <typename T>
hid_t HDF5Handler::createDataset(std::string datasetName, std::vector<hsize_t> gridDims, std::vector<hsize_t> arrayDims) {                
    if (usingMPI) {MPI_Barrier(comm);}
    hid_t dset_id = createHDF5GridOfArrays<T>(file_id, datasetName, gridDims, arrayDims);
    currentlyOpenDatasets.insert(std::pair<std::string, hid_t>(datasetName, dset_id));
    if (usingMPI) {MPI_Barrier(comm);}
    return dset_id;
}
template hid_t HDF5Handler::createDataset<float>(std::string datasetName, std::vector<hsize_t> gridDims, std::vector<hsize_t> arrayDims);
template hid_t HDF5Handler::createDataset<double>(std::string datasetName, std::vector<hsize_t> gridDims, std::vector<hsize_t> arrayDims);
template hid_t HDF5Handler::createDataset<hdf_complex_t>(std::string datasetName, std::vector<hsize_t> gridDims, std::vector<hsize_t> arrayDims);
template hid_t HDF5Handler::createDataset<unsigned short>(std::string datasetName, std::vector<hsize_t> gridDims, std::vector<hsize_t> arrayDims);

void HDF5Handler::openDataset(std::string datasetName) {                
    if (currentlyOpenDatasets.count(datasetName) == 0) {
        if (usingMPI) {MPI_Barrier(comm);}
        hid_t dset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
        currentlyOpenDatasets.insert(std::pair<std::string, hid_t>(datasetName, dset_id));
        if (usingMPI) {MPI_Barrier(comm);}
    }
    else {
        std::cerr << "Warning: attempted to open dataset " << datasetName << " that was already open." << std::endl;
    }
}

std::vector<std::string> HDF5Handler::getDatasetNames() {
    hsize_t numObjs;
    HDFCHECK(H5Gget_num_objs(this->file_id, &numObjs));
    std::vector<std::string> names;
    for (unsigned int i=0; i<numObjs; i++) {
        int type = H5Gget_objtype_by_idx(this->file_id, i);
        HDFCHECK(type);
        if (type == H5G_DATASET) {
            ssize_t nameLength = H5Gget_objname_by_idx(this->file_id, i, NULL, 0);
            char *name = new char[nameLength];
            H5Gget_objname_by_idx(this->file_id, i, name, nameLength+1);
            names.push_back(std::string(name, name+nameLength));
            delete[] name;
        }
    }
    return names;
}

HDF5Handler::~HDF5Handler() {
    // Close datasets
    for (auto&& nameIDPair : currentlyOpenDatasets) {
        HDFCHECK(H5Dclose(nameIDPair.second));
    }
    currentlyOpenDatasets.clear();
    
    // Close property lists.
    HDFCHECK(H5Pclose(this->plist_id_file_access));
    HDFCHECK(H5Pclose(this->plist_id_xfer_independent));
    
    // Close file handle
    HDFCHECK(H5Fclose(this->file_id));
}



} //END NAMESPACE HPP