/** @file testTensor.cpp
* @author Michael Malahe
* @brief Tests for functions in tensor.h.
* @details Does not include behaviour limited to individual tensor orders. Those
* can be found in testTensor2.cpp and testTensor4.cpp;
*/

#include "mpi.h"
#include <hpp/mpiUtils.h>
#include <stdexcept>
#include <cassert>
#include <iostream>

namespace hpp
{

/**
 * @brief The tests that don't require any parallel processing.
 */
void testMPIUtilsNonParallel() {    
    
    // Recognition of existing types
    if (MPIType<double>() !=  MPI_DOUBLE) throw std::runtime_error("Type recognition failed.");
    if (MPIType<float>() !=  MPI_FLOAT) throw std::runtime_error("Type recognition failed.");
    
    // Failure on incompatible type
    bool caughtTypeIncompatible = false;
    try {        
        MPIType<void>();
    } catch (std::runtime_error& e) {
        caughtTypeIncompatible = true;
    }
    if (!caughtTypeIncompatible)  throw std::runtime_error("Non-existent MPI type not raised correctly.");
}

void testMPIUtilsParallel(MPI_Comm comm, int comm_size, int comm_rank) {
    // Min
    int val = comm_rank+1;
    int minVal = MPIMin(val, comm);
    if (minVal != 1) throw std::runtime_error("Min failed.");
    val = comm_size-1-comm_rank+1;
    minVal = MPIMin(val, comm);
    if (minVal != 1) throw std::runtime_error("Min failed.");
    
    // Max
    val = comm_rank+1;
    int maxVal = MPIMax(val, comm);
    if (maxVal != comm_size) throw std::runtime_error("Max failed.");
    val = comm_size-comm_rank;
    maxVal = MPIMax(val, comm);
    if (maxVal != comm_size) throw std::runtime_error("Max failed.");
    
    // Sum
    val = comm_rank+1;
    int analyticSum = ((comm_size)*(comm_size+1))/2;
    int sum = MPISum(val, comm);
    if (sum != analyticSum) throw std::runtime_error("Sum failed.");
    
    // True
    bool condition = true;
    bool allTrue = MPIAllTrue(condition, comm);
    if (!allTrue) throw std::runtime_error("allTrue failed.");
    if (comm_rank == 0) condition=false;
    allTrue = MPIAllTrue(condition, comm);
    if (allTrue) throw std::runtime_error("allTrue failed.");
}

}//END NAMESPACE HPP

int main(int argc, char *argv[]) {
    // Init
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Non parallel tests
    if (comm_rank == 0) {
        hpp::testMPIUtilsNonParallel();
    }
    
    // Parallel tests
    MPI_Barrier(comm);
    hpp::testMPIUtilsParallel(comm, comm_size, comm_rank);
    
    // Finalize
    MPI_Finalize();
    
    // Return
    return 0;
}