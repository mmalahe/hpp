/// @file testTensor2.cpp
/// @author Michael Malahe
/// @brief Code for testing members of hpp::Tensor2
#include <hpp/tensor.h>
#include <cassert>
#include <iostream>
#include <limits>
#include "mpi.h"
#include <hdf5/serial/H5Cpp.h>
#include <hdf5/openmpi/hdf5.h>
#include <hpp/mpiUtils.h>
#include <hpp/hdfUtils.h>

namespace hpp {

/**
 * @brief Test the basic functions of Tensor2
 * @tparam the scalar type
 */
template <typename T>
void testTensor2Parallel(MPI_Comm comm, int comm_size, int comm_rank) {
    // Summation
    int dim = 2;
    T localVal = (T)comm_rank+1;
    Tensor2<T> localTensor = localVal*ones2<T>(dim);
    T analyticSum = ((comm_size)*(comm_size+1))/2;
    Tensor2<T> analyticSumTensor = analyticSum*ones2<T>(dim);
    Tensor2<T> sumTensor = MPISum(localTensor, comm);
    if (analyticSumTensor != sumTensor) throw std::runtime_error("Summation failed.");
}

/**
 * @brief In this test we write to and read from an HDF5 dataset in parallel
 * @details Note that there isn't a CPP interface to the parallel I/O, so
 * the CPP interface is used for the serial parts, and the C interface is
 * used for the parallel parts.
 * @param comm The MPI communicator to use
 * @param comm_size The size of the communicator
 * @param comm_rank The rank of this process on the communicator
 */
template <typename T>
void testTensor2ParallelIO(MPI_Comm comm, int comm_size, int comm_rank) {        
    // File and dataset names
    H5std_string filename = "parallelOutput.h5";
    H5std_string datasetName = "tensorArray";
    
    // Create an arbitrarily sized two dimensional arrays of tensors
    std::vector<hsize_t> tensorGridDims = {2,3};
    std::vector<hsize_t> tensorDims = {5,7};
    
    // Distribute work by linear index
    unsigned int nGridpoints = tensorGridDims[0]*tensorGridDims[1];
    unsigned int nGridpointsLocal = nGridpoints/comm_size;
    if (nGridpoints % comm_size != 0) nGridpointsLocal++;
    unsigned int gridStart = comm_rank*nGridpointsLocal;
    unsigned int gridEnd = gridStart+nGridpointsLocal-1;
    if (gridEnd > nGridpoints-1) gridEnd = nGridpoints-1;
    printf("Proc %d taking care of %ud-%ud.\n", comm_rank, gridStart, gridEnd);
    
    // Create file and dataset //
    /////////////////////////////
    
    HDF5MPIHandler file(filename, comm, true);
    file.createDataset<T>(datasetName, tensorGridDims, tensorDims);    
    hid_t dset_main = file.getDataset(datasetName);
    hid_t plist_id_xfer_independent = file.getPropertyListTransferIndependent();
    
    // Write to dataset in parallel using C API //
    //////////////////////////////////////////////
    
    // Write to file on each process
    for (unsigned int flatIdx=gridStart; flatIdx<gridEnd+1; flatIdx++) {
        // Get grid coordinates from  linear index
        idx2d gridIdx = hpp::unflat(flatIdx, tensorGridDims[0], tensorGridDims[1]);
        std::vector<hsize_t> arrayOffset = {gridIdx.i, gridIdx.j};
        
        // Create output tensor
        Tensor2<T> outputTensor(tensorDims[0], tensorDims[1]);
        for (unsigned int i=0; i<tensorDims[0]; i++) {
            for (unsigned int j=0; j<tensorDims[1]; j++) {
                // Make sure that this expression is also matched in verification below
                outputTensor(i,j) = 10*i + j + comm_rank;
            }
        }
        
        // Write tensor
        outputTensor.writeToExistingHDF5Dataset(dset_main, plist_id_xfer_independent, arrayOffset);
    }
    
    // Read back in from dataset in parallel using C API //
    ///////////////////////////////////////////////////////
    
    // Read from file on each process
    for (unsigned int flatIdx=gridStart; flatIdx<gridEnd+1; flatIdx++) {
        // Get grid coordinates from  linear index
        idx2d gridIdx = hpp::unflat(flatIdx, tensorGridDims[0], tensorGridDims[1]);
        std::vector<hsize_t> arrayOffset = {gridIdx.i, gridIdx.j};
        
        // Create output tensor
        Tensor2<T> outputTensor(tensorDims[0], tensorDims[1]);
        for (unsigned int i=0; i<tensorDims[0]; i++) {
            for (unsigned int j=0; j<tensorDims[1]; j++) {
                // Make sure that this expression is also matched in output step above
                outputTensor(i,j) = 10*i + j + comm_rank;
            }
        }
        
        // Check against input tensor
        Tensor2<T> inputTensor(dset_main, plist_id_xfer_independent, arrayOffset, tensorDims);
        if (inputTensor != outputTensor) {
            throw TensorError("Tensor written isn't the same as tensor read.");
        }
    }
}

} //END NAMESPACE HPP

int main(int argc, char *argv[]) {
    // Init
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Parallel tests
    hpp::testTensor2Parallel<double>(comm, comm_size, comm_rank); 
    hpp::testTensor2ParallelIO<double>(comm, comm_size, comm_rank);  
    
    // Finalize
    MPI_Finalize();
    
    // Return
    return 0;
}

