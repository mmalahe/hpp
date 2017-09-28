/* @file mpiUtils.h
* @author Michael Malahe
* @brief Header file for MPI helper functions
* @details
*/

#ifndef HPP_MPIUTILS_H
#define HPP_MPIUTILS_H

#include "mpi.h"
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <hpp/config.h>

namespace hpp
{

/**
 * @brief Get the MPI datatype from the C++ type
 * @return The MPI_Datatype
 */
template <typename T>
MPI_Datatype MPIType() {
    MPI_Datatype datatype;
    if (std::is_same<T, double>::value) {
        datatype = MPI_DOUBLE;
    }
    else if (std::is_same<T, float>::value) {
        datatype = MPI_FLOAT;
    }
    else if (std::is_same<T, int>::value) {
        datatype = MPI_INT;
    }
    else if (std::is_same<T, unsigned int>::value) {
        datatype = MPI_UNSIGNED;
    }
    else if (std::is_same<T, long>::value) {
        datatype = MPI_LONG;
    }
    else if (std::is_same<T, unsigned long>::value) {
        datatype = MPI_UNSIGNED_LONG;
    }
    else {
        throw std::runtime_error("Type not recognised.");
    }
    return datatype;
}

template <typename T>
T MPIMin(const T& localVal, MPI_Comm comm) {
    T globalVal;
    MPI_Allreduce(&localVal, &globalVal, 1, MPIType<T>(), MPI_MIN, comm);
    return globalVal;
}

template <typename T>
T MPIMax(const T& localVal, MPI_Comm comm) {
    T globalVal;
    MPI_Allreduce(&localVal, &globalVal, 1, MPIType<T>(), MPI_MAX, comm);
    return globalVal;
}

template <typename T> 
T MPISum(const T& localVal, MPI_Comm comm) {
    T globalVal;
    MPI_Allreduce(&localVal, &globalVal, 1, MPIType<T>(), MPI_SUM, comm);
    return globalVal;
}

template <typename T> 
std::vector<T> MPIConcatOnRoot(T localVal, MPI_Comm comm) {
    // Comm
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Gather
    std::vector<T> globalVec;
    if (comm_rank == 0) {
        globalVec.resize(comm_size);
    }
    MPI_Datatype dataType = MPIType<T>();
    MPI_Gather(&localVal, 1, dataType, globalVec.data(), 1, dataType, 0, comm);
    
    // Return
    return globalVec;
}

template <typename T> 
std::vector<T> MPIConcatOnRoot(std::vector<T> localVec, MPI_Comm comm) {
    // Comm
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Let root know how many values are coming from each process
    int globalRecvCountSize = 0;
    if (comm_rank == 0) {
        globalRecvCountSize = comm_size;
    }
    std::vector<int> recvcounts(globalRecvCountSize);
    int nLocal = localVec.size();
    MPI_Gather(&nLocal, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, comm);
    
    // Get displacements and total size
    int globalVecSize = 0;
    std::vector<int> displs;
    if (comm_rank == 0) {
        displs.push_back(0);
        for (int i=0; i<comm_size-1; i++) {
            globalVecSize += recvcounts[i];
            displs.push_back(globalVecSize);
        }
        globalVecSize += recvcounts[comm_size-1];
    }
    
    // Gather the global vector on root
    std::vector<T> globalVec(globalVecSize);
    MPI_Datatype dataType = MPIType<T>();
    MPI_Gatherv(localVec.data(), nLocal, dataType, globalVec.data(), recvcounts.data(),
    displs.data(), dataType, 0, comm);
    
    // Return
    return globalVec;
}

template <typename T> 
T MPIBroadcastFromRoot(T rootVal, MPI_Comm comm) {
    // Comm
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);    
    
    // Scatter value
    T localVal = rootVal;
    MPI_Datatype dataType = MPIType<T>();
    MPI_Bcast(&localVal, 1, dataType, 0, comm);
    
    // Return
    return localVal;
}

template <typename T> 
std::vector<T> MPIBroadcastFromRoot(std::vector<T> rootVec, MPI_Comm comm) {
    // Comm
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Get size of root vector
    // Size can be anything on non-root vectors
    unsigned int size = rootVec.size();
    size = MPIBroadcastFromRoot(size, comm);
    
    // Create local vector
    std::vector<T> localVec(size);
    if (comm_rank == 0) {
        localVec = rootVec;
    }
    
    // Scatter values
    MPI_Datatype dataType = MPIType<T>();
    MPI_Bcast(localVec.data(), size, dataType, 0, comm);
    
    // Return
    return localVec;
}

template <typename T> 
std::vector<T> MPISplitVectorEvenly(const std::vector<T>& rootVec, MPI_Comm comm) {
    // Comm
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Sizing of local vectors and MPI parameters
    int rootVecSize = -1;
    int localVecSize = -1;
    int localVecSizeFinalProc = -1;
    std::vector<int> sendcounts(comm_size);
    std::vector<int> displs(comm_size);
    if (comm_rank == 0) {
        // Divide up vector
        rootVecSize = rootVec.size();
        localVecSize = rootVecSize/comm_size;
        localVecSizeFinalProc = rootVecSize-localVecSize*(comm_size-1);        
        
        // Create MPI parameters
        displs[0] = 0;
        for (int i=0; i<comm_size-1; i++) {
            sendcounts[i] = localVecSize;
            displs[i+1] = displs[i] + localVecSize;
        }
        sendcounts.back() = localVecSizeFinalProc;
    }
    
    // Broadcast parameters from root
    rootVecSize = MPIBroadcastFromRoot(rootVecSize, comm);
    localVecSize = MPIBroadcastFromRoot(localVecSize, comm);
    localVecSizeFinalProc = MPIBroadcastFromRoot(localVecSizeFinalProc, comm);
    sendcounts = MPIBroadcastFromRoot(sendcounts, comm);
    displs = MPIBroadcastFromRoot(displs, comm);
    
    // Final processor may have a different size
    if (comm_rank == comm_size-1) {
        localVecSize = localVecSizeFinalProc;
    }
    
    // Scatter to local vectors
    std::vector<T> localVec(localVecSize);
    MPI_Scatterv(rootVec.data(), sendcounts.data(), displs.data(), MPIType<T>(),
                 localVec.data(), localVecSize, MPIType<T>(), 0, comm);
                 
    // Return
    return localVec;
}

template <typename T> 
std::vector<T> MPISplitVectorEvenly(const std::vector<T>& rootVec, MPI_Comm comm, MPI_Datatype dtype) {
    // Comm
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    
    // Sizing of local vectors and MPI parameters
    int rootVecSize = -1;
    int localVecSize = -1;
    int localVecSizeFinalProc = -1;
    std::vector<int> sendcounts(comm_size);
    std::vector<int> displs(comm_size);
    if (comm_rank == 0) {
        // Divide up vector
        rootVecSize = rootVec.size();
        localVecSize = rootVecSize/comm_size;
        localVecSizeFinalProc = rootVecSize-localVecSize*(comm_size-1);        
        
        // Create MPI parameters
        displs[0] = 0;
        for (int i=0; i<comm_size-1; i++) {
            sendcounts[i] = localVecSize;
            displs[i+1] = displs[i] + localVecSize;
        }
        sendcounts.back() = localVecSizeFinalProc;
    }
    
    // Broadcast parameters from root
    rootVecSize = MPIBroadcastFromRoot(rootVecSize, comm);
    localVecSize = MPIBroadcastFromRoot(localVecSize, comm);
    localVecSizeFinalProc = MPIBroadcastFromRoot(localVecSizeFinalProc, comm);
    sendcounts = MPIBroadcastFromRoot(sendcounts, comm);
    displs = MPIBroadcastFromRoot(displs, comm);
    
    // Final processor may have a different size
    if (comm_rank == comm_size-1) {
        localVecSize = localVecSizeFinalProc;
    }
    
    // Scatter to local vectors
    std::vector<T> localVec(localVecSize);
    MPI_Scatterv(rootVec.data(), sendcounts.data(), displs.data(), dtype,
                 localVec.data(), localVecSize, dtype, 0, comm);
                 
    // Return
    return localVec;
}

/**
 * @brief Determine if condition is true for all processes
 * @param condition
 * @param comm
 * @return True if true for all, false otherwise
 */
bool MPIAllTrue(bool condition, MPI_Comm comm);

}//END NAMESPACE HPP

#endif /* HPP_MPIUTILS_H */