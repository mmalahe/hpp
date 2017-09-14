/** @file testHDFUtilsParallel.cpp
* @author Michael Malahe
* @brief Tests for parallel functions in hdfUtils.h
*/

#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>
#include <hpp/hdfUtils.h>
#include <hpp/tensor.h>

namespace hpp {

void testHDFReadWriteParallel(MPI_Comm comm) {    
    // MPI config
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // File
    std::string filename = "parallelFile.hdf5";
    HDF5MPIHandler file(filename, comm, true);
    
    // Common grid for datasets
    std::vector<hsize_t> gridDims = {2,3};
    
    // Dataset 1: containing a grid of scalars
    std::string dataset1Name = "dataset1";
    std::vector<hsize_t> dataset1Dims = gridDims;
    file.createDataset<float>(dataset1Name, dataset1Dims);
    hid_t dataset1 = file.getDataset(dataset1Name);
    
    // Dataset 2: containing a grid of vectors
    std::string dataset2Name = "dataset2";
    std::vector<hsize_t> dataset2ArrayDims = {5};
    std::vector<hsize_t> dataset2Dims = gridDims;
    dataset2Dims.insert(std::end(dataset2Dims), std::begin(dataset2ArrayDims), std::end(dataset2ArrayDims));
    file.createDataset<float>(dataset2Name, dataset2Dims);
    hid_t dataset2 = file.getDataset(dataset2Name);
    
    // Transfer property list
    hid_t plist_id_xfer_independent = file.getPropertyListTransferIndependent();
    
    // Distribute work by linear index
    unsigned int nGridpoints = gridDims[0]*gridDims[1];
    unsigned int nGridpointsLocal = nGridpoints/comm_size;
    if (nGridpoints % comm_size != 0) nGridpointsLocal++;
    unsigned int gridStart = comm_rank*nGridpointsLocal;
    unsigned int gridEnd = gridStart+nGridpointsLocal-1;
    if (gridEnd > nGridpoints-1) gridEnd = nGridpoints-1;
    printf("Proc %d taking care of %u-%u.\n", comm_rank, gridStart, gridEnd);
    
    // Write data
    for (unsigned int flatIdx=gridStart; flatIdx<gridEnd+1; flatIdx++) {
        // Common grid offset for datasets
        idx2d gridIdx = unflat(flatIdx, gridDims[0], gridDims[1]);
        std::vector<hsize_t> gridOffset = {gridIdx.i, gridIdx.j};
        
        // Dataset 1
        float d1value = (float)flatIdx;
        writeSingleHDF5Value(dataset1, plist_id_xfer_independent, gridOffset, &d1value);
        
        // Dataset 2
        std::vector<float> d2values(dataset2ArrayDims[0]);
        for (unsigned int i=0; i<dataset2ArrayDims[0]; i++) {
            d2values[i] = (float)flatIdx + 0.1*i;
        }
        writeSingleHDF5Array(dataset2, plist_id_xfer_independent, gridOffset, dataset2ArrayDims, d2values.data());
    }
    
    // Read data
    for (unsigned int flatIdx=gridStart; flatIdx<gridEnd+1; flatIdx++) {
        // Common grid offset for datasets
        idx2d gridIdx = unflat(flatIdx, gridDims[0], gridDims[1]);
        std::vector<hsize_t> gridOffset = {gridIdx.i, gridIdx.j};
        
        // Dataset 1
        float d1valueOut = (float)flatIdx;
        float d1valueIn;
        readSingleHDF5Value(dataset1, plist_id_xfer_independent, gridOffset, &d1valueIn);
        if (d1valueIn != d1valueOut) throw std::runtime_error("Dataset 1 value in is not value out.");
        
        // Dataset 2
        std::vector<float> d2valuesOut(dataset2ArrayDims[0]);
        for (unsigned int i=0; i<dataset2ArrayDims[0]; i++) {
            d2valuesOut[i] = (float)flatIdx + 0.1*i;
        }
        std::vector<float> d2valuesIn(dataset2ArrayDims[0]);
        readSingleHDF5Array(dataset2, plist_id_xfer_independent, gridOffset, dataset2ArrayDims, d2valuesIn.data());
        if (d2valuesIn != d2valuesOut) throw std::runtime_error("Dataset 2 value in is not value out.");
    }
}

} //END NAMESPACE HPP

int main(int argc, char *argv[]) {
    // Init
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    
    // Parallel tests
    hpp::testHDFReadWriteParallel(comm);
    
    // Finalize
    MPI_Finalize();
    
    // Return
    return 0;
}