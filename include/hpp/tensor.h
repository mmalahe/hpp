/// @file tensor.h
/// @author Michael Malahe
/// @brief Header file for tensor classes
#ifndef HPP_TENSOR_H
#define HPP_TENSOR_H

#include <cstddef>
#include <iostream>
#include <cassert>
#include <lapacke.h>
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <valarray>
#include <random>
#include <hdf5/serial/H5Cpp.h>
#include <hdf5/openmpi/hdf5.h>
#include <hpp/config.h>
#include <hpp/mpiUtils.h>
#include <hpp/hdfUtilsCpp.h>
#include <hpp/hdfUtils.h>
#include <unsupported/Eigen/MatrixFunctions>
#include "mpi.h"

#define HPP_ARRAY_LAYOUT LAPACK_ROW_MAJOR

#ifdef DEBUG_BUILD
    #define DEBUG_ONLY(x) x
#else
    #define DEBUG_ONLY(x)
#endif

namespace hpp
{

// SOME OVERLOADS OF STD::VECTOR

/**
 * @brief Elementwise division
 * @param vec
 * @param scalar
 * @return 
 */
template <typename T>
std::vector<T> operator/(const std::vector<T>& vec, T scalar) {
    std::vector<T> newvec = vec;
    for (auto&& v : newvec) {
        v /= scalar;
    }
    return newvec;
}

template <typename T>
std::vector<std::vector<T>> operator/(const std::vector<std::vector<T>>& veclist, T scalar) {
    std::vector<std::vector<T>> newveclist = veclist;
    for (auto&& vec : newveclist) {
        for (auto&& v : vec) {
            v /= scalar;
        }
    }
    return newveclist;
}

template <typename T>
void operator*=(std::vector<T>& vec, const T scalar) {
    for (auto&& v : vec) {
        v *= scalar;
    }
}

template <typename T>
void operator/=(std::vector<T>& vec, const T scalar) {
    for (auto&& v : vec) {
        v /= scalar;
    }
}

template <typename T>
std::vector<T> operator*(const std::vector<T>& vec, const T scalar) {
    std::vector<T> newvec = vec;
    newvec *= scalar;
    return newvec;
}

template <typename T>
std::vector<T> operator*(const T scalar, const std::vector<T>& vec) {
    return vec*scalar;
}

template <typename T>
std::vector<std::vector<T>> operator*(T scalar, const std::vector<std::vector<T>>& veclist) {
    return veclist*scalar;
}

template <typename T>
std::vector<T> abs(const std::vector<T>& vec) {
    std::vector<T> absVec = vec;
    for (auto&& v : absVec) {
        v = std::abs(v);
    }
    return absVec;
}

template <typename T>
T min(const std::vector<T>& vec) {
    T val = *(std::min_element(vec.begin(), vec.end()));
    return val;
}

template <typename T>
T max(const std::vector<T>& vec) {
    T val = *(std::max_element(vec.begin(), vec.end()));
    return val;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    unsigned int n = vec1.size();
    if (n != vec2.size()) {
        throw std::runtime_error("Vector size mismatch.");
    }
    std::vector<T> vecout(n);
    for (unsigned int i=0; i<n; i++) { 
        vecout[i] = vec1[i] - vec2[i];
    }
    return vecout;
}

/**
 * @brief Product of all the entries of a vector
 * @param vec the vector
 * @return the product
 */
template <typename T>
T prod(const std::vector<T>& vec) {
    T product = 1.0;
    for (auto&& v : vec) {
        product *= v;
    }
    return product;
}

//
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec)
{
    out << "[";
    for (unsigned int i=0; i<vec.size(); i++) {
        out << vec[i];
        if (i != vec.size()-1) {
            out << ", ";
        }
    }
    out << "]";
    return out;
}
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::valarray<T>& vec)
{
    out << "[";
    for (unsigned int i=0; i<vec.size(); i++) {
        out << vec[i];
        if (i != vec.size()-1) {
            out << ", ";
        }
    }
    out << "]";
    return out;
}  

// TENSORS

class TensorError: public std::runtime_error
{
    public:
        explicit TensorError (const std::string &val) : std::runtime_error::runtime_error(val) {}
};

/**
 * @struct idx2d
 * @brief A 2D index
 * @var idx2d::i
 * the first index
 * @var idx2d::j 
 * the second index
 */
struct idx2d{
    unsigned int i;
    unsigned int j;
};

inline idx2d unflat(unsigned int flatIdx, unsigned int n1, unsigned int n2) 
{
    idx2d idx;
    if (HPP_ARRAY_LAYOUT == LAPACK_ROW_MAJOR) {
        idx.i = flatIdx/n2;
        idx.j = flatIdx - idx.i*n2;
    } else if (HPP_ARRAY_LAYOUT == LAPACK_COL_MAJOR) {
        idx.j = flatIdx/n1;
        idx.i = flatIdx - idx.j*n1;
    } else {
        throw TensorError(std::string("No support for layout ")+std::to_string(HPP_ARRAY_LAYOUT));
    }
    return idx;
}

/**
 * @struct idx4d
 * @brief A 4D index
 * @var idx4d::i 
 * the first index
 * @var idx4d::j 
 * the second index
 * @var idx4d::k 
 * the third index
 * @var idx4d::l 
 * the fourth index
 */
struct idx4d{
    unsigned int i;
    unsigned int j;
    unsigned int k;
    unsigned int l;
};

inline idx4d unflat(unsigned int flatIdx, unsigned int n1, unsigned int n2,
            unsigned int n3, unsigned int n4)
{
    idx4d idx;
    if (HPP_ARRAY_LAYOUT == LAPACK_ROW_MAJOR) {
        idx.i = flatIdx/(n2*n3*n4);
        idx.j = (flatIdx - idx.i*n2*n3*n4)/(n3*n4);
        idx.k = (flatIdx - idx.i*n2*n3*n4 - idx.j*n3*n4)/(n4);
        idx.l = (flatIdx - idx.i*n2*n3*n4 - idx.j*n3*n4 - idx.k*n4);
    } else if (HPP_ARRAY_LAYOUT == LAPACK_COL_MAJOR) {
        idx.i = flatIdx/(n3*n2*n1);
        idx.j = (flatIdx - idx.i*n3*n2*n1)/(n2*n1);
        idx.k = (flatIdx - idx.i*n3*n2*n1 - idx.j*n2*n1)/(n1);
        idx.l = (flatIdx - idx.i*n3*n2*n1 - idx.j*n2*n1 - idx.k*n1);
    }
    else {
        throw TensorError(std::string("No support for layout ")+std::to_string(HPP_ARRAY_LAYOUT));
    }
    return idx;
}

template <typename T, typename U>
std::vector<T> unflatC(T flatIdx, std::vector<U> dims) {
    // Dimension of the array
    unsigned int rank = dims.size();
    
    // Array index to be returned
    std::vector<T> idx(rank);
    
    // Remaining total of the flat index to account for
    T remainingIdx = flatIdx;
    
    // The size of the stride at the current level
    unsigned int stride = 1;
    for (unsigned int i=0; i<rank; i++) {
        stride *= dims[i];
    }
    
    // Get the index
    for (unsigned int i=0; i<rank; i++) {
        // Stride is smaller at the next level
        stride /= dims[i];
        
        // Get the index
        idx[i] = remainingIdx/stride;
        
        // Subtract from the remaining index
        remainingIdx -= idx[i]*stride;
    }
    
    // Sanity check
    if (remainingIdx != 0) {
        throw std::runtime_error("Something has gone wrong with the index calculation.");
    }
    
    // Return
    return idx;
}

template <typename T, typename U>
T flatC(std::vector<T> idx, std::vector<U> dims) {
    unsigned int rank = dims.size();
    if (idx.size() != rank) throw TensorError("Dimensions mismatch.");
    T flatIdx = 0;
    unsigned int stride = 1;
    for (int i=rank-1; i>=0; i--) {
        flatIdx += idx[i]*stride;
        stride *= dims[i];
    }
    return flatIdx;
}

// Additional functions for std::vector
template <typename T>
std::vector<T> ones(unsigned int n) {
    std::vector<T> vec(n);
    for (auto&& v : vec) {
        v = 1.0;
    }
    return vec;
}

// Forward declarations are necessarry for binary operations
template <typename T>
class Tensor2; 
template <typename T>
class Tensor4;

// CLASSES AND STRUCTS WITH TENSORS AS COMPONENTS
template <typename T>
struct PolarDecomposition{
    Tensor2<T> R;
    Tensor2<T> U;
};

// TENSOR 2D //
///////////////

/**
 * @class Tensor2
 * @brief A class for second order tensors
 * @tparam T the scalar type
 */
template <typename T>
class Tensor2
{
    friend Tensor4<T>;
    template<typename U, unsigned int M, unsigned int N> friend class Tensor2CUDA;
    
    public:
        // Default constructor
        Tensor2();
    
        // Basic zeroing constructor
        Tensor2(const unsigned int n1, const unsigned int n2);
        
        // Copy constructor
        Tensor2(const Tensor2<T> &A);
        
        // Getters
        unsigned int getn1() const {return n1;}
        unsigned int getn2() const {return n2;}
        unsigned int getNVals() const {return nVals;}
        
        // Get read-only value by two indices
        T getVal(const unsigned int i, const unsigned int j) const;
        
        // Get read-only value by single flat index
        T getValFlat(const unsigned int flatIdx) const;
        
        // Get read/write reference to value
        void setVal(const unsigned int i, const unsigned int j, T val);
        T &operator()(const unsigned int i, const unsigned int j);
        T &operator()(const unsigned int flatIdx);
        void copyValuesOut(T* outVals) const;
        
        // Construct from Tensor4
        explicit Tensor2(const Tensor4<T>& A);

        // Assignment operator
        Tensor2<T>& operator=(const Tensor2<T>& input);

        // Printing
        void printToStream(std::ostream& out) const;
        
        // HDF5 I/O, C++ interface
        Tensor2(H5::DataSet& dataset, std::vector<hsize_t> gridOffset, std::vector<hsize_t> tensorDims);
        void writeToExistingHDF5Dataset(H5::DataSet& dataset, std::vector<hsize_t> arrayOffset);
        
        Tensor2(H5::H5File& file, const H5std_string& datasetName, std::vector<hsize_t> tensorDims);
        void writeToNewHDF5(const H5std_string& filename, const H5std_string& datasetName); 
        void addAsNewHDF5Dataset(H5::H5File& file, const H5std_string& datasetName);     

        // HDF5 I/O, C interface
        Tensor2(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> gridOffset, std::vector<hsize_t> tensorDims);
        void writeToExistingHDF5Dataset(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> arrayOffset);
        
        // Inverse
        void invInPlace();
        Tensor2<T> inv() const;

        // Exponential
        Tensor2<T> exp() const;
        
        // Assert squareness
        void assertSquare() const;
        
        // Check if rotation matrix
        bool isRotationMatrix() const;
        
        // Basic properties
        T tr() const;
        T det() const;
        T min() const;
        T max() const;
        T absmax() const;
        T spectralNorm() const;
        T frobeniusNorm() const;
        
        // Product of all elements;
        T prod() const;
        
        // Basic conversions
        Tensor2<T> abs() const;
        Tensor2<T> trans() const;
        
        // Constrain values
        Tensor2<T> constrainedTo(const T minVal, const T maxVal) const;
        void constrainInPlace(const T minVal, const T maxVal);
        Tensor2<T> scaledToUnitDeterminant() const;
        
        // Deviatoric component
        Tensor2<T> deviatoricComponent() const;
        
        // Decompositions
        void evecDecomposition(std::valarray<T>& evals, Tensor2<T>& evecs);
        Tensor2<T> sqrtMatrix();
        PolarDecomposition<T> polarDecomposition() const;
        
        // FRIENDS
        
        // Dimension matching
        template <typename U>
        friend bool areSameShape(const Tensor2<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend void assertSameShape(const Tensor2<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend void assertCompatibleForMultiplication(const Tensor2<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend void assertCompatibleForContraction(const Tensor4<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend void assertCompatibleForContraction(const Tensor4<U>& A, const Tensor2<U>& B, const Tensor2<U>& C);
        
        // Equality
        template <typename U>
        friend bool areEqual(const Tensor2<U>& A, const Tensor2<U>& B, U tol);
        template <typename U>
        friend bool operator==(const Tensor2<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend bool operator!=(const Tensor2<U>& A, const Tensor2<U>& B);
        
        // External operators
        template <typename U>
        friend Tensor2<U> operator*(const Tensor2<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend void ABPlusBTransposeAInPlace(const hpp::Tensor2<U>& A, const hpp::Tensor2<U>& B, hpp::Tensor2<U>& C);
        template <typename U>
        friend Tensor2<U> contract(const Tensor4<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend U contract(const Tensor2<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend void contractInPlace(const Tensor4<U>& A, const Tensor2<U>& B, Tensor2<U>& C);
        template <typename U>
        friend void assertCompatibleForOuterProduct(const Tensor2<U>& A, const Tensor2<U>& B, const Tensor4<U>& C);
        template <typename U>
        friend Tensor4<U> outer(const Tensor2<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend void outerInPlace(const Tensor2<U>& A, const Tensor2<U>& B, Tensor4<U>& C);
        
        // Addition
        template <typename U>
        friend Tensor2<U> operator+(const Tensor2<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend Tensor2<U> operator+(const Tensor2<U>&A, const U&B);
        template <typename U>
        friend Tensor2<U> operator+(const U&A, const Tensor2<U>& B);
        template <typename U>
        friend void operator+=(Tensor2<U>& A, const U& B);
        template <typename U>
        friend void operator+=(Tensor2<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend Tensor2<U> MPISum(Tensor2<U>& local, MPI_Comm comm);
        
        // Subtraction
        template <typename U>
        friend Tensor2<U> operator-(const Tensor2<U>& A, const Tensor2<U>& B);
        template <typename U>
        friend Tensor2<U> operator-(const Tensor2<U>& A, const U& B);
        template <typename U>
        friend Tensor2<U> operator-(const U&A, const Tensor2<U>& B);
        template <typename U>
        friend void operator-=(Tensor2<U>& A, const U& B);
        template <typename U>
        friend void operator-=(Tensor2<U>& A, const Tensor2<U>& B);
        
        // Multiplication
        template <typename U>
        friend Tensor2<U> operator*(const Tensor2<U>&A, const U &B);
        template <typename U>
        friend Tensor2<U> operator*(const U&A, const Tensor2<U>& B);
        template <typename U>
        friend void operator*=(Tensor2<U>& A, const U& B);
        
        // Division
        template <typename U>
        friend Tensor2<U> operator/(const Tensor2<U>&A, const U &B);
        template <typename U>
        friend void operator/=(Tensor2<U>& A, const U& B);
        
        // Special tensors
        template <typename U>
        friend void identityTensor2InPlace(unsigned int n, Tensor2<U>& A);

    protected:
        /** @brief the first dimension of the tensor */
        unsigned int n1 = 0; 
        /** @brief the second dimension of the tensor */
        unsigned int n2 = 0;
        /** @brief the total number of elements in the tensor */
        unsigned int nVals = 0;
        
         /** @brief the underlying 1D array */
        std::valarray<T> vals;

    private:         
        // Initialization
        void initialize(const unsigned int n1, const unsigned int n2);
        
        // Copy values from another tensor
        void copyValues(const Tensor2<T>& input);
        
        // HDF I/O C++ API
        void constructFromHDF5Dataset(H5::DataSet& dataset, std::vector<hsize_t> gridOffset, std::vector<hsize_t> tensorDims);
        
        // HDF I/O C API
        void constructFromHDF5Dataset(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> gridOffset, std::vector<hsize_t> tensorDims);
        
        // Construct from existing data
        Tensor2(const unsigned int n1, const unsigned int n2, const T* inArray);
        Tensor2(const unsigned int n1, const unsigned int n2, const std::valarray<T>& inVec);
        
        // Flattening and unflattening indices
        unsigned int flat(const unsigned int i, const unsigned int j) const;
        idx2d unflat(const unsigned int idx) const;
};

// INLINE MEMBER FUNCTIONS //

/**
 * @brief Flatten the two indices of the tensor
 * @details The flattening order is dictated by the macro HPP_ARRAY_LAYOUT
 * @param i the first index
 * @param j the second index
 * @return the flattened index
 */
template <typename T>
inline unsigned int Tensor2<T>::flat(const unsigned int i, const unsigned int j) const
{
    if (HPP_ARRAY_LAYOUT == LAPACK_ROW_MAJOR) {
        return i*n2 + j;
    } else if (HPP_ARRAY_LAYOUT == LAPACK_COL_MAJOR) {
        return i + j*n1;
    } else {
        throw TensorError(std::string("No support for layout ")+std::to_string(HPP_ARRAY_LAYOUT));
    }
}

/**
 * @brief Unflatten the two indices of the tensor
 * @details The unflattening order is dictated by the macro HPP_ARRAY_LAYOUT
 * @param flat_idx the flattened index
 * @return an idx2d with the unflattened indices
 */
template <typename T>
inline idx2d Tensor2<T>::unflat(const unsigned int flat_idx) const
{
    idx2d idx;
    if (HPP_ARRAY_LAYOUT == LAPACK_ROW_MAJOR) {
        idx.i = flat_idx/n2;
        idx.j = flat_idx - idx.i*n2;
    } else if (HPP_ARRAY_LAYOUT == LAPACK_COL_MAJOR) {
        idx.j = flat_idx/n1;
        idx.i = flat_idx - idx.j*n1;
    } else {
        throw TensorError(std::string("No support for layout ")+std::to_string(HPP_ARRAY_LAYOUT));
    }
    
    return idx;
}

/**
 * @brief Get the value of \f$T_{ij}\f$
 * @details Bounds checking is done for debug builds
 * @param i
 * @param j
 * @tparam T the scalar type
 * @return \f$T_{ij}\f$
 */
template <typename T>
inline T Tensor2<T>::getVal(unsigned int i, unsigned int j) const
{
    unsigned int flatIdx = this->flat(i,j);
    return vals[flatIdx];
}

/**
 * @brief Get the value of \f$T_{i}\f$
 * @details Where \f$i\f$ is the index in the underlying array.
 * Bounds checking is done for debug builds. 
 * @param flatIdx i
 * @tparam T the scalar type
 * @return \f$T_{i}\f$
 * @return 
*/
template <typename T>
inline T Tensor2<T>::getValFlat(unsigned int flatIdx) const
{
 return vals[flatIdx];
}

/**
 * @brief Get the a reference to the value of \f$T_{ij}\f$
 * @details Bounds checking is done for debug builds
 * @param i
 * @param j
 * @tparam T the scalar type
 * @return \f$T_{ij}\f$
 */
template <typename T>
inline T& Tensor2<T>::operator()(const unsigned int i, const unsigned int j)
{
    unsigned int flatIdx = this->flat(i,j);
    return vals[flatIdx];
}

template <typename T>
inline T& Tensor2<T>::operator()(const unsigned int flatIdx)
{
    return vals[flatIdx];
}

template <typename T>
void Tensor2<T>::setVal(const unsigned int i, const unsigned int j, T val) {
    unsigned int flatIdx = this->flat(i,j);
    vals[flatIdx] = val;
}

////

// Outer product with std::vector
template <typename T>
Tensor2<T> outer(const std::vector<T>& A, const std::vector<T>& B) {
    int n1 = A.size();
    int n2 = B.size();
    Tensor2<T> C(n1,n2);
    for (int i=0; i<n1; i++) {
        for (int j=0; j<n2; j++) {
            C(i,j) = A[i]*B[j];
        }
    }
    return C;
}

template <typename T>
void outerInPlace(const std::vector<T>& A, const std::vector<T>& B, Tensor2<T>& C) {
    unsigned int n1 = A.size();
    unsigned int n2 = B.size();
    DEBUG_ONLY(if (n1 != C.getn1() || n2 != C.getn2()) throw TensorError("Size mismatch."););
    for (unsigned int i=0; i<n1; i++) {
        for (unsigned int j=0; j<n2; j++) {
            C(i,j) = A[i]*B[j];
        }
    }
}

// Outer product with std::valarray
template <typename T>
Tensor2<T> outer(const std::valarray<T>& A, const std::valarray<T>& B) {
    int n1 = A.size();
    int n2 = B.size();
    Tensor2<T> C(n1,n2);
    for (int i=0; i<n1; i++) {
        for (int j=0; j<n2; j++) {
            C(i,j) = A[i]*B[j];
        }
    }
    return C;
}

// Basic tensors
template<typename T>
Tensor2<T> identityTensor2(unsigned int n)
{
    Tensor2<T> A = Tensor2<T>(n,n);
    for (unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) {
            A(i,j) = i==j;
        }
    }
    return A;
}

template<typename T>
void identityTensor2InPlace(unsigned int n, Tensor2<T>& A)
{
    if (A.getn1() != n || A.getn2() != n) {
        throw std::runtime_error("Wrong dimensions.");
    }
    A.vals = (T)0.0;
    for (unsigned int i=0; i<n; i++) {
        A(i,i) = (T)1.0;
    }
}

template<typename T>
Tensor2<T> ones2(unsigned int n)
{
    Tensor2<T> A = Tensor2<T>(n,n);
    for (unsigned int i=0; i<A.getNVals(); i++) {
        A(i) = 1.0;
    }
    return A;
}

// Basic tensors
template<typename T>
Tensor2<T> diag2(const std::vector<T>& vals)
{
    unsigned int n = vals.size();
    Tensor2<T> A(n,n);
    for (unsigned int i=0; i<n; i++) {
        A(i,i) = vals[i];
    }
    return A;
}

template<typename T>
Tensor2<T> diag2(const std::valarray<T>& vals)
{
    unsigned int n = vals.size();
    Tensor2<T> A(n,n);
    for (unsigned int i=0; i<n; i++) {
        A(i,i) = vals[i];
    }
    return A;
}

template<typename T>
bool areEqual(const Tensor2<T>& A, const Tensor2<T>& B, T tol) {
    if (A.n1 != B.n1) return false;
    if (A.n2 != B.n2) return false;
    bool areEq = true;
    for (unsigned int i=0; i<A.getNVals(); i++) {
        T diff = A.getValFlat(i)-B.getValFlat(i);
        if (std::abs(diff) > tol) {
            areEq = false;
            break;
        }
    }
    return areEq;
}

/**
 * @brief Test equality of two tensors
 * @details Uses the arbitrary \f$10 \epsilon_{\mathrm{machine}}\f$ to determine 
 * if floats are equal. Test coverage in testTensor2Basics().
 * @param A first tensor
 * @param B second tensor
 * @tparam T the scalar type
 * @return true if they're equal, false if they're not
 */
template <typename T>
bool operator==(const Tensor2<T>& A, const Tensor2<T>& B) {
    return areEqual(A, B, (T)(10.0*std::numeric_limits<T>::epsilon()));
}

// Inequality
template <typename T>
bool operator!=(const Tensor2<T>& A, const Tensor2<T>& B) {
    return !(A==B);
}

// Compatability
template <typename T>
inline bool areSameShape(const Tensor2<T>& A, const Tensor2<T>& B) {
    return (A.n1 == B.n1 && A.n2 == B.n2);
}

template <typename T>
void assertSameShape(const Tensor2<T>& A, const Tensor2<T>& B) {
    if (!areSameShape(A,B)) {
        throw TensorError(std::string("Dimension mismatch."));
    }
}

// ADDITION

/**
 * @brief Addition
 * @param A first tensor
 * @param B second tensor
 * @tparam T the scalar type
 * @return A+b
 */
template <typename T>
Tensor2<T> operator+(const Tensor2<T>& A, const Tensor2<T>& B) {
    assertSameShape(A,B);
    Tensor2<T> C = A;
    C.vals += B.vals;
    return C;
}

template <typename T>
void operator+=(Tensor2<T>& A, const Tensor2<T>& B) {
    A.vals += B.vals;
}

template <typename T>
Tensor2<T> operator+(const Tensor2<T>& A, const T& B) {
    Tensor2<T> C = A;
    C.vals += B;
    return C;
}

template <typename T>
Tensor2<T> operator+(const T&A, const Tensor2<T>& B) {
    return B+A;
}

template <typename T>
void operator+=(Tensor2<T>& A, const T& B) {
    A.vals += B;
}

template <typename T>
Tensor2<T> MPISum(Tensor2<T>& local, MPI_Comm comm) {
    Tensor2<T> global(local.getn1(), local.getn2());
    MPI_Allreduce(&(local.vals[0]), &(global.vals[0]), local.getNVals(), MPIType<T>(), MPI_SUM, comm);
    return global;
}

// Subtraction
template <typename T>
Tensor2<T> operator-(const Tensor2<T>& A, const Tensor2<T>& B) {
    assertSameShape(A,B);
    Tensor2<T> C = A;
    C.vals -= B.vals;
    return C;
}

/**
 * @brief 
 * @param A
 * @param B
 * @tparam T the scalar type
 */
template <typename T>
void operator-=(Tensor2<T>& A, const Tensor2<T>& B) {
    A.vals -= B.vals;
}

template <typename T>
Tensor2<T> operator-(const Tensor2<T>& A, const T& B) {
    Tensor2<T> C = A;
    C.vals -= B;
    return C;
}

template <typename T>
Tensor2<T> operator-(const T&A, const Tensor2<T>& B) {
    return B-A;
}

template <typename T>
void operator-=(Tensor2<T>& A, const T& B) {
    A.vals -= B;
}
// Multiplication
// Note: we haven't implemented element-wise multiplication with another tensor,
// because it would conflict with normal matrix multiplication

template <typename T>
Tensor2<T> operator*(const Tensor2<T>& A, const T& B) {
    Tensor2<T> C = A;
    C.vals *= B;
    return C;
}

template <typename T>
Tensor2<T> operator*(const T&A, const Tensor2<T>& B) {
    return B*A;
}

template <typename T>
void operator*=(Tensor2<T>& A, const T& B) {
    A.vals *= B;
}

/* Division
Note: we haven't implemented element-wise division with another tensor,
or scalar divided by tensor, because the meaning of those operators has a lot 
of room for ambiguity (thanks MATLAB)
*/
template <typename T>
Tensor2<T> operator/(const Tensor2<T>& A, const T& B) {
    Tensor2<T> C = A;
    C.vals /= B;
    return C;
}

template <typename T>
void operator/=(Tensor2<T>& A, const T& B) {
    A.vals /= B;
}

template <typename T>
void assertCompatibleForMultiplication(const Tensor2<T>& A, const Tensor2<T>& B) {
    if (A.n2 != B.n1) {
        throw TensorError("Shapes are incompatible for multiplication.");
    }
}

// Tensor multiplication
template<typename T>
Tensor2<T> operator*(const Tensor2<T>& A, const Tensor2<T>& B)
{
    assertCompatibleForMultiplication(A,B);
    unsigned int m = A.n1;
    unsigned int n = A.n2;
    unsigned int p = B.n2;
    Tensor2<T> C = Tensor2<T>(m,p);
    for (unsigned int i=0; i<m; i++) {
        for (unsigned int j=0; j<p; j++) {
            for (unsigned int k=0; k<n; k++) {
                C(i,j) += A.getVal(i,k)*B.getVal(k,j);
            }
        }
    }
    return C;
}

//
template <typename T>
void ABPlusBTransposeAInPlace(const hpp::Tensor2<T>& A, const hpp::Tensor2<T>& B, hpp::Tensor2<T>& C) {
    A.assertSquare();
    assertSameShape(A,B);
    assertSameShape(B,C);
    unsigned int m = A.n1;
    C.vals = (T)0.0;
    for (unsigned int i=0; i<m; i++) {
        for (unsigned int j=0; j<m; j++) {
            for (unsigned int k=0; k<m; k++) {
                C(i,j) += A.getVal(i,k)*B.getVal(k,j);
                C(i,j) += B.getVal(k,i)*A.getVal(k,j);
            }
        }
    }
}

// Tensor contraction
template<typename T>
T contract(const Tensor2<T>& A, const Tensor2<T>& B)
{
    assertSameShape(A,B);
    T C = 0;
    for (unsigned int i=0; i<A.getNVals(); i++) {
        C += A.getValFlat(i)*B.getValFlat(i);
    }
    return C;
}

// Stream output
template <typename T>
std::ostream& operator<<(std::ostream& out, const Tensor2<T>& A)
{
    A.printToStream(out);
    return out;
}


// TENSOR 4D //
///////////////

/**
 * @class Tensor4
 * @author Michael Malahe
 * @date 07/10/16
 * @file tensor.h
 * @brief This class will do no bounds-checking or any other safety checks that
 * induce significant overhead.
 */
template <typename T>
class Tensor4
{
    friend Tensor2<T>;
    template<typename U, unsigned int M, unsigned int N, unsigned int P, unsigned int Q> friend class Tensor4CUDA;
    
public:
    // Default constructor
    Tensor4();

    // Constructor and desctructor
    Tensor4(unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4);
    ~Tensor4();

    // Construct from 2nd order
    explicit Tensor4(const Tensor2<T>& A);
    
    // Assignment operator
    Tensor4<T>& operator=(const Tensor4<T>& input);

    // Copy constructor
    Tensor4(const Tensor4<T> &input);
    
    // Getters
    unsigned int getNVals() const {return nVals;}
    
    // Read/write access
    T& operator()(unsigned int i, unsigned int j, unsigned int k, unsigned int l);
    T& operator()(unsigned int flatIdx);
    
    // Read only access
    T getVal(unsigned int i, unsigned int j, unsigned int k, unsigned int l) const;
    T getValFlat(unsigned int flatIdx) const;
    
    // Basic properties
    T frobeniusNorm() const;
    
    // Inverse
    void invInPlace();
    Tensor4<T> inv() const;
    
    // Printing
    void printToStream(std::ostream& out);
    
    // Getters    
    unsigned int getn1() const {return n1;}
    unsigned int getn2() const {return n2;}
    unsigned int getn3() const {return n3;}
    unsigned int getn4() const {return n4;}

    // EXTERNAL FRIEND FUNCTIONS
    
    // Equality
    template <typename U>
    friend bool areSameShape(const Tensor4<U>& A, const Tensor4<U>& B);
    template <typename U>
    friend void assertSameShape(const Tensor4<U>& A, const Tensor4<U>& B);
    template <typename U>
    friend bool operator==(const Tensor4<U>& A, const Tensor4<U>& B);
    template <typename U>
    friend bool operator!=(const Tensor4<U>& A, const Tensor4<U>& B);
    
    // Contraction
    template <typename U>
    friend void assertCompatibleForContraction(const Tensor4<U>& A, const Tensor2<U>& B);
    template <typename U>
    friend void assertCompatibleForContraction(const Tensor4<U>& A, const Tensor2<U>& B, const Tensor2<U>& C);
    template <typename U>
    friend Tensor2<U> contract(const Tensor4<U>& A, const Tensor2<U>& B);
    template <typename U>
    friend void contractInPlace(const Tensor4<U>& A, const Tensor2<U>& B, Tensor2<U>& C);
    template <typename U>
    friend void assertCompatibleForContraction(const Tensor4<U>& A, const Tensor4<U>& B);
    template <typename U>
    friend Tensor4<U> contract(const Tensor4<U>& A, const Tensor4<U>& B);
    
    // Products
    template <typename U>
    friend void assertCompatibleForOuterProduct(const Tensor2<U>& A, const Tensor2<U>& B, const Tensor4<U>& C);
    template <typename U>
    friend void outerInPlace(const Tensor2<U>& A, const Tensor2<U>& B, Tensor4<U>& C);
    
    // Addition
    template <typename U>
    friend Tensor4<U> operator+(const Tensor4<U>& A, const Tensor4<U>& B);
    template <typename U>
    friend Tensor4<U> operator+(const Tensor4<U>&A, const U&B);
    template <typename U>
    friend Tensor4<U> operator+(const U&A, const Tensor4<U>& B);
    template <typename U>
    friend void operator+=(Tensor4<U>& A, const U& B);
    template <typename U>
    friend void operator+=(Tensor4<U>& A, const Tensor4<U>& B);
    
    // Subtraction
    template <typename U>
    friend Tensor4<U> operator-(const Tensor4<U>& A);
    template <typename U>
    friend Tensor4<U> operator-(const Tensor4<U>& A, const Tensor4<U>& B);
    template <typename U>
    friend Tensor4<U> operator-(const Tensor4<U>& A, const U& B);
    template <typename U>
    friend Tensor4<U> operator-(const U&A, const Tensor4<U>& B);
    template <typename U>
    friend void operator-=(Tensor4<U>& A, const U& B);
    template <typename U>
    friend void operator-=(Tensor4<U>& A, const Tensor4<U>& B);
    
    // Multiplication
    template <typename U>
    friend Tensor4<U> operator*(const Tensor4<U>&A, const U &B);
    template <typename U>
    friend Tensor4<U> operator*(const U&A, const Tensor4<U>& B);
    template <typename U>
    friend void operator*=(Tensor4<U>& A, const U& B);
    
    // Division
    template <typename U>
    friend Tensor4<U> operator/(const Tensor4<U>&A, const U &B);
    template <typename U>
    friend void operator/=(Tensor4<U>& A, const U& B);

protected:
    unsigned int n1 = 0;
    unsigned int n2 = 0;
    unsigned int n3 = 0;
    unsigned int n4 = 0;
    unsigned int nVals = 0;
    
    // Underlying contiguous array
    std::valarray<T> vals;

private:

    // Initialization
    void initialize(unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4);

    // Flattening and unflattening indices
    unsigned int flat(unsigned int i, unsigned int j, unsigned int k, unsigned int l) const;
    idx4d unflat(unsigned int idx);
    
    void copyValues(const Tensor4<T>& input);
};

// INLINE MEMBER FUNCTIONS //

/**
 * @brief Flatten the four indices of the tensor
 *
 * This is currently done in row-major order.
 * @param i the first index
 * @param j the second index
 * @param k the thrid index
 * @param l the fourth index
 * @return the flattened index
 */
template <typename T>
inline unsigned int Tensor4<T>::flat(unsigned int i, unsigned int j, unsigned int k, unsigned int l) const
{
    return i*n2*n3*n4 + j*n3*n4 + k*n4 + l;
}

/**
 * @brief Unflatten the four indices of the tensor
 * @details This is currently done in row-major order.
 * @param flat_idx idx the flattened index
 * @return an idx4d instance with the unflattened indices
 */
template <typename T>
inline idx4d Tensor4<T>::unflat(unsigned int flat_idx)
{
    idx4d idx;
    idx.i = flat_idx/(n2*n3*n4);
    idx.j = (flat_idx - idx.i*n2*n3*n4)/(n3*n4);
    idx.k = (flat_idx - idx.i*n2*n3*n4 - idx.j*n3*n4)/(n4);
    idx.l = (flat_idx - idx.i*n2*n3*n4 - idx.j*n3*n4 - idx.k*n4);
    return idx;
}

// Get values
template <typename T>
inline T Tensor4<T>::getVal(unsigned int i, unsigned int j, unsigned int k, unsigned int l) const
{
    unsigned int flatIdx = this->flat(i,j,k,l);
    return vals[flatIdx];
}

// Get values
template <typename T>
inline T Tensor4<T>::getValFlat(unsigned int flatIdx) const
{
    return vals[flatIdx];
}

template <typename T>
inline T& Tensor4<T>::operator()(unsigned int i, unsigned int j, unsigned int k, unsigned int l)
{
    unsigned int flatIdx = this->flat(i,j,k,l);
    return vals[flatIdx];
}

template <typename T>
inline T& Tensor4<T>::operator()(unsigned int flatIdx)
{
    return vals[flatIdx];
}

/////

// Stream output
template <typename T>
std::ostream& operator<<(std::ostream& out, Tensor4<T>& A)
{
    A.printToStream(out);
    return out;
}

// Basic tensors
template<typename T>
inline Tensor4<T> identityTensor4(unsigned int n)
{
    Tensor4<T> A = Tensor4<T>(n,n,n,n);
    for (unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) {
            for (unsigned int k=0; k<n; k++) {
                for (unsigned int l=0; l<n; l++) {
                    A(i,j,k,l) = (i==k)*(j==l);
                }
            }
        }
    }
    return A;
}

template<typename T>
void identityTensor4InPlace(unsigned int n, Tensor4<T>& A)
{
    if (A.getn1() != n || A.getn2() != n || A.getn3() != n || A.getn4() != n) {
        throw std::runtime_error("Wrong dimensions.");
    }
    for (unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) {
            for (unsigned int k=0; k<n; k++) {
                for (unsigned int l=0; l<n; l++) {
                    A(i,j,k,l) = (i==k)*(j==l);
                }
            }
        }
    }
}

// Compatability
template <typename T>
inline bool areSameShape(const Tensor4<T>& A, const Tensor4<T>& B) {
    return (A.n1 == B.n1 && A.n2 == B.n2 && A.n3 == B.n3 && A.n4 == B.n4);
}

template <typename T>
void assertSameShape(const Tensor4<T>& A, const Tensor4<T>& B) {
    if (!areSameShape(A,B)) {
        throw TensorError(std::string("Dimension mismatch."));
    }
}

/**
 * @brief Test equality of two tensors
 * @details Uses the arbitrary \f$10 \epsilon_{\mathrm{machine}}\f$ to determine 
 * if floats are equal. Test coverage in testTensor4Basics().
 * @param A first tensor
 * @param B second tensor
 * @tparam T the scalar type
 * @return true if they're equal, false if they're not
 */
template <typename T>
bool operator==(const Tensor4<T>& A, const Tensor4<T>& B) {
    if (A.n1 != B.n1) return false;
    if (A.n2 != B.n2) return false;
    if (A.n3 != B.n3) return false;
    if (A.n4 != B.n4) return false;
    bool areEqual = true;
    for (unsigned int i=0; i<A.getNVals(); i++) {
        T diff = A.getValFlat(i)-B.getValFlat(i);
        if (std::abs(diff) > 10.0*std::numeric_limits<T>::epsilon()) {
            areEqual = false;
            break;
        }
    }
    return areEqual;
}

// Inequality
template <typename T>
bool operator!=(const Tensor4<T>& A, const Tensor4<T>& B) {
    return !(A==B);
}

/**
 * @brief Addition
 * @param A first tensor
 * @param B second tensor
 * @tparam T the scalar type
 * @return A+B
 */
template <typename T>
Tensor4<T> operator+(const Tensor4<T>& A, const Tensor4<T>& B) {
    assertSameShape(A,B);
    Tensor4<T> C = A;
    C.vals += B.vals;
    return C;
}

template <typename T>
void operator+=(Tensor4<T>& A, const Tensor4<T>& B) {
    A.vals += B.vals;
}

template <typename T>
Tensor4<T> operator+(const Tensor4<T>& A, const T& B) {
    Tensor4<T> C = A;
    C.vals += B;
    return C;
}

template <typename T>
Tensor4<T> operator+(const T&A, const Tensor4<T>& B) {
    return B+A;
}

template <typename T>
void operator+=(Tensor4<T>& A, const T& B) {
    A.vals += B;
}

// Unary negation
template <typename T>
Tensor4<T> operator-(const Tensor4<T>& A) {
    Tensor4<T> C = A;
    C.vals *= -1.0;
    return C;
}

// Subtraction
template <typename T>
Tensor4<T> operator-(const Tensor4<T>& A, const Tensor4<T>& B) {
    assertSameShape(A,B);
    Tensor4<T> C = A;
    C.vals -= B.vals;
    return C;
}

/**
 * @brief 
 * @param A
 * @param B
 * @tparam T the scalar type
 */
template <typename T>
void operator-=(Tensor4<T>& A, const Tensor4<T>& B) {
    A.vals -= B.vals;
}

template <typename T>
Tensor4<T> operator-(const Tensor4<T>& A, const T& B) {
    Tensor4<T> C = A;
    C.vals -= B;
    return C;
}

template <typename T>
Tensor4<T> operator-(const T&A, const Tensor4<T>& B) {
    return B-A;
}

template <typename T>
void operator-=(Tensor4<T>& A, const T& B) {
    A.vals -= B;
}
// Multiplication
// Note: we haven't implemented element-wise multiplication with another tensor,
// because it would conflict with normal matrix multiplication

template <typename T>
Tensor4<T> operator*(const Tensor4<T>& A, const T& B) {
    Tensor4<T> C = A;
    C.vals *= B;
    return C;
}

template <typename T>
Tensor4<T> operator*(const T&A, const Tensor4<T>& B) {
    return B*A;
}

template <typename T>
void operator*=(Tensor4<T>& A, const T& B) {
    A.vals *= B;
}

/* Division
Note: we haven't implemented element-wise division with another tensor,
or scalar divided by tensor, because the meaning of those operators has a lot 
of room for ambiguity (thanks MATLAB)
*/
template <typename T>
Tensor4<T> operator/(const Tensor4<T>& A, const T& B) {
    Tensor4<T> C = A;
    C.vals /= B;
    return C;
}

template <typename T>
void operator/=(Tensor4<T>& A, const T& B) {
    A.vals /= B;
}

// TENSOR 2 AND TENSOR 4 INTERACTIONS

// Outer product
template<typename T>
Tensor4<T> outer(const Tensor2<T>& A, const Tensor2<T>& B)
{
    unsigned int m = A.n1;
    unsigned int n = A.n2;
    unsigned int p = B.n1;
    unsigned int q = B.n2;
    Tensor4<T> C(m,n,p,q);
    for (unsigned int i=0; i<m; i++) {
        for (unsigned int j=0; j<n; j++) {
            for (unsigned int k=0; k<p; k++) {
                for (unsigned int l=0; l<q; l++) {
                    C(i,j,k,l) = A.getVal(i,j)*B.getVal(k,l);
                }
            }
        }
    }
    return C;
}

template <typename T>
void assertCompatibleForOuterProduct(const Tensor2<T>& A, const Tensor2<T>& B, const Tensor4<T>& C) {
    if (A.n1 != C.n1 || A.n2 != C.n2 || B.n1 != C.n3 || B.n2 != C.n4) {
        throw TensorError("Shapes are incompatible for contraction.");
    }
}

template<typename T>
void outerInPlace(const Tensor2<T>& A, const Tensor2<T>& B, Tensor4<T>& C)
{
    unsigned int m = A.n1;
    unsigned int n = A.n2;
    unsigned int p = B.n1;
    unsigned int q = B.n2;
    assertCompatibleForOuterProduct(A, B, C);
    for (unsigned int i=0; i<m; i++) {
        for (unsigned int j=0; j<n; j++) {
            for (unsigned int k=0; k<p; k++) {
                for (unsigned int l=0; l<q; l++) {
                    C(i,j,k,l) = A.getVal(i,j)*B.getVal(k,l);
                }
            }
        }
    }
}

//
template <typename T>
inline void assertCompatibleForContraction(const Tensor4<T>& A, const Tensor2<T>& B) {
    if (A.n3 != B.n1 || A.n4 != B.n2) {
        throw TensorError("Shapes are incompatible for contraction.");
    }
}

template <typename T>
inline void assertCompatibleForContraction(const Tensor4<T>& A, const Tensor2<T>& B, const Tensor2<T>& C) {
    if (A.n1 != C.n1 || A.n2 != C.n2 || A.n3 != B.n1 || A.n4 != B.n2) {
        throw TensorError("Shapes are incompatible for contraction.");
    }
}

// Tensor contraction
template <typename T>
inline Tensor2<T> contract(const Tensor4<T>& A, const Tensor2<T>& B)
{
    assertCompatibleForContraction(A,B);
    unsigned int m = A.n1;
    unsigned int n = A.n2;
    unsigned int p = A.n3;
    unsigned int q = A.n4;
    Tensor2<T> C = Tensor2<T>(m,n);
    for (unsigned int i=0; i<m; i++) {
        for (unsigned int j=0; j<n; j++) {
            for (unsigned int k=0; k<p; k++) {
                for (unsigned int l=0; l<q; l++) {
                    C(i,j) += A.getVal(i,j,k,l)*B.getVal(k,l);
                }
            }
        }
    }
    return C;
}

template <typename T>
inline void contractInPlace(const Tensor4<T>& A, const Tensor2<T>& B, Tensor2<T>& C)
{
    assertCompatibleForContraction(A,B,C);
    unsigned int m = A.n1;
    unsigned int n = A.n2;
    unsigned int p = A.n3;
    unsigned int q = A.n4;
    C.vals = (T)0.0;
    for (unsigned int i=0; i<m; i++) {
        for (unsigned int j=0; j<n; j++) {
            for (unsigned int k=0; k<p; k++) {
                for (unsigned int l=0; l<q; l++) {
                    C(i,j) += A.getVal(i,j,k,l)*B.getVal(k,l);
                }
            }
        }
    }
}

template <typename U>
inline void assertCompatibleForContraction(const Tensor4<U>& A, const Tensor4<U>& B) {
    if (A.n3 != B.n1 || A.n4 != B.n2) {
        throw TensorError("Shapes are incompatible for contraction.");
    }
}

template <typename U>
inline Tensor4<U> contract(const Tensor4<U>& A, const Tensor4<U>& B) {
    assertCompatibleForContraction(A,B);
    unsigned int m = A.n1;
    unsigned int n = A.n2;
    unsigned int p = A.n3;
    unsigned int q = A.n4;
    unsigned int r = B.n3;
    unsigned int s = B.n4;
    Tensor4<U> C(m,n,r,s);
    for (unsigned int im=0; im<m; im++) {
        for (unsigned int in=0; in<n; in++) {
            for (unsigned int ir=0; ir<r; ir++) {
                for (unsigned int is=0; is<s; is++) {
                    C(im,in,ir,is) = 0.0;
                    for (unsigned int ip=0; ip<p; ip++) {
                        for (unsigned int iq=0; iq<q; iq++) {
                            C(im,in,ir,is) += A.getVal(im,in,ip,iq)*B.getVal(ip,iq,ir,is);
                        }
                    }
                }
            }
        }
    }
    return C;
}

// Interactions with std::vector
template <typename T>
std::vector<T> operator*(const hpp::Tensor2<T>& A, const std::vector<T>& x)
{
    unsigned int m = A.getn1();
    unsigned int n = A.getn2();
    if (n != x.size()) {
        throw hpp::TensorError("Incompatible tensor vector multiplication sizes.");
    }
    std::vector<T> b(m);
    for (unsigned int i=0; i<m; i++) {
        for (unsigned int j=0; j<n; j++) {
            b[i] += A.getVal(i,j)*x[j];
        }
    }
    return b;
}

// Interactions with std::valarray
template <typename T>
std::valarray<T> operator*(const hpp::Tensor2<T>& A, const std::valarray<T>& x)
{
    unsigned int m = A.getn1();
    unsigned int n = A.getn2();
    if (n != x.size()) {
        throw hpp::TensorError("Incompatible tensor vector multiplication sizes.");
    }
    std::valarray<T> b(m);
    for (unsigned int i=0; i<m; i++) {
        for (unsigned int j=0; j<n; j++) {
            b[i] += A.getVal(i,j)*x[j];
        }
    }
    return b;
}

// Interactions with std::vector
/**
 * @brief Replicates the numpy behaviour of vec*mat
 * @param x the vector
 * @param A the matrix
 * @return x*A
 */
template <typename T>
hpp::Tensor2<T> operator*(const std::vector<T>& x, const hpp::Tensor2<T>& A)
{
    unsigned int m = A.getn1();
    unsigned int n = A.getn2();
    if (n != x.size()) {
        throw hpp::TensorError("Incompatible vector tensor multiplication sizes.");
    }
    hpp::Tensor2<T> B = A;
    for (unsigned int i=0; i<m; i++) {
        for (unsigned int j=0; j<n; j++) {
            B(i,j) *= x[j];
        }
    }
    return B;
}

// TRANSFORMATIONS
inline std::mt19937 makeMt19937(bool defaultSeed) {
    if (defaultSeed) {
        return std::mt19937();
    }
    else {
        std::random_device rd;
        return std::mt19937(rd());
    }
}

// Arvo 1992 method
template <typename T>
void rotationTensorFrom3UniformRandomsArvo1992(Tensor2<T>& A, T x1, T x2, T x3) {    
    // Static to avoid expensive alloc/dealloc for the tens/hundreds of millions regime
    static std::vector<T> v(3);
    static Tensor2<T> R(3,3);
    static Tensor2<T> H(3,3);
    
    // Random rotation about vertical axis
    R(0,0) = std::cos(2*M_PI*x1);
    R(1,1) = R(0,0);
    R(0,1) = std::sin(2*M_PI*x1);
    R(1,0) = -R(0,1);
    R(2,2) = 1.0;
    
    // Random rotation of vertical axis    
    v[0] = std::cos(2*M_PI*x2)*std::sqrt(x3);
    v[1] = std::sin(2*M_PI*x2)*std::sqrt(x3);
    v[2] = std::sqrt(1-x3);
    
    // Householder reflection
    outerInPlace(v, v, H);
    H *= (T)2.0;
    H(0,0) -= 1.0;
    H(1,1) -= 1.0;
    H(2,2) -= 1.0;
    
    // Final matrix
    A = H*R;
}

template <typename T>
void randomRotationTensorArvo1992(std::mt19937& gen, Tensor2<T>& A) {    
    // Static to avoid expensive alloc/dealloc for the tens/hundreds of millions regime
    static std::vector<T> v(3);
    static Tensor2<T> R(3,3);
    static Tensor2<T> H(3,3);
    
    // double is specified here, as if it is not, the values out of the
    // distribution are different for float and double, even with the same
    // default seed
    static std::uniform_real_distribution<double> dist(0.0,1.0);
    
    // Random samples
    T x1 = (T)dist(gen);
    T x2 = (T)dist(gen);
    T x3 = (T)dist(gen);
    
    // Generate 
    rotationTensorFrom3UniformRandomsArvo1992(A, x1, x2, x3);
}

template <typename T>
void randomRotationTensorArvo1992(Tensor2<T>& A, bool defaultSeed=false) {
    // Create generator once
    static std::mt19937 gen = makeMt19937(defaultSeed);
    
    // Generate
    randomRotationTensorArvo1992(gen, A);
}

template <typename T>
void rotationTensorFrom3UniformRandoms(Tensor2<T>& A, T x1, T x2, T x3) {
    rotationTensorFrom3UniformRandomsArvo1992(A, x1, x2, x3);
} 

/**
 * @brief Generate a random rotation tensor.
 * @details This is translated from scipy/linalg/tests/test_decomp.py
 * The description there, verbatim is:
 * Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(n)).
    The algorithm is described in the paper
    Stewart, G.W., 'The efficient generation of random orthogonal
    matrices with an application to condition estimators', SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization
 * @param dim
 * @param defaultSeed if true, use a default random seed, otherwise generate a truly random one
 * @return The rotation tensor
 */
template <typename T>
Tensor2<T> randomRotationTensorStewart1980(unsigned int dim, bool defaultSeed=false) {
    // Generator
    static std::mt19937 gen = makeMt19937(defaultSeed);
    
    // Normal distribution to draw from
    std::normal_distribution<T> dist(0.0,1.0);   
    
    // Construction
    Tensor2<T> H = identityTensor2<T>(dim);
    std::vector<T> D = ones<T>(dim);
    for (unsigned int n=1; n<dim; n++) {
        std::valarray<T> x(dim-n+1);
        for (unsigned int i=0; i<dim-n+1; i++) {
            x[i] = dist(gen);
        }
        D[n-1] = std::copysign(1.0, x[0]);
        x[0] -= D[n-1]*std::sqrt((x*x).sum());
        
        // Householder transformation
        Tensor2<T> Hx = identityTensor2<T>(dim-n+1) - (T)2.0*outer(x, x)/(x*x).sum();
        Tensor2<T> mat = identityTensor2<T>(dim);
        for (unsigned int i=n-1; i<dim; i++) {
            for (unsigned int j=n-1; j<dim; j++) {
                mat(i,j) = Hx(i-n+1,j-n+1);
            }
        }
        H = H*mat;
    }
    
    // Fix the last sign such that the determinant is 1
    D[dim-1] = -prod(D);
    H = (D*H.trans()).trans();
    
    // Return
    return H;
}

/**
 * @brief Generate a random rotation tensor.
 * @details This is translated from scipy/linalg/tests/test_decomp.py
 * The description there, verbatim is:
 * Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(n)).
    The algorithm is described in the paper
    Stewart, G.W., 'The efficient generation of random orthogonal
    matrices with an application to condition estimators', SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization
 * @param dim
 * @param A the output tensor to populate
 * @param defaultSeed if true, use a default random seed, otherwise generate a truly random one
 */
template <typename T>
void randomRotationTensorInPlace(unsigned int dim, Tensor2<T>& A, bool defaultSeed=false) {
    // Generate tensor
    if (dim == 3) {
        randomRotationTensorArvo1992<T>(A, defaultSeed);
    }
    else {
        A = randomRotationTensorStewart1980<T>(dim, defaultSeed);
    }
    
    // Check the tensor
    #ifdef DEBUG_BUILD
        if (!(A.isRotationMatrix())) {
            std::cerr << "Warning: random rotation tensor didn't pass check. Re-generating." << std::endl;
            randomRotationTensorInPlace<T>(dim, A, defaultSeed);
        }
    #endif
}

template <typename T>
Tensor2<T> randomRotationTensor(unsigned int dim, bool defaultSeed=false) {
    // Generate tensor
    Tensor2<T> A(dim,dim);
    randomRotationTensorInPlace<T>(dim, A, defaultSeed);
    return A;
}

template <typename T>
Tensor2<T> EulerZXZRotationMatrix(T alpha, T beta, T gamma) {
    Tensor2<T> R(3,3);
    T c1 = std::cos(alpha);
    T c2 = std::cos(beta);
    T c3 = std::cos(gamma);
    T s1 = std::sin(alpha);
    T s2 = std::sin(beta);
    T s3 = std::sin(gamma);
    R(0,0) = c1*c3 - c2*s1*s3;
    R(0,1) = -c1*s3 - c2*c3*s1;
    R(0,2) = s1*s2;
    R(1,0) = c3*s1 + c1*c2*s3;
    R(1,1) = c1*c2*c3 - s1*s3;
    R(1,2) = -c1*s2;
    R(2,0) = s2*s3;
    R(2,1) = c3*s2;
    R(2,2) = c2;
    return R;
}

/**
 * @class EulerAngles
 * @author Michael Malahe
 * @date 28/08/17
 * @file tensor.h
 * @brief Euler Angles
 * @details Defined according to the following conventions:
 * - Right-handed
 * - Counter-clockwise
 * - Active
 * - Extrinsic
 * - Z-X-Z sequence
 * - \f$\alpha\f$: The second applied Z rotation angle, \f$[0,2\pi) \f$
 * - \f$\beta\f$: The X rotation angle, \f$[0,\pi) \f$
 * - \f$\gamma\f$: The first applied Z rotation angle, \f$[0,2\pi) \f$
 * - The resulting rotation matrix is the product 
 * \f$ R = Z(\alpha) X(\beta) Z(\gamma) \f$,
 * Where Z is the elemental rotation matrix about z, and X is the elemental
 * rotation matrix about X. 
 */
template <typename T>
struct EulerAngles {
    T alpha = 0;
    T beta = 0;
    T gamma = 0;
    
    // Getters/setters (mainly intended for Python interface)
    T getAlpha() const {return alpha;}
    T getBeta() const {return beta;}
    T getGamma() const {return gamma;}
    void setAlpha(const T& alpha) {this->alpha = alpha;}
    void setBeta(const T& beta) {this->beta = beta;}
    void setGamma(const T& gamma) {this->gamma = gamma;}
};

template <typename T>
MPI_Datatype getEulerAnglesTypeMPI() {
    MPI_Datatype dtype;
    MPI_Type_contiguous(3, MPIType<T>(), &dtype);
    MPI_Type_commit(&dtype);
    return dtype;
}

/**
 * @brief Convert polar angles to Euler angles
 * @param theta the azimuthal angle
 * @param phi the zenithal angle
 * @return The Euler angles
 */
template <typename T>
EulerAngles<T> polarToEuler(T theta, T phi) {
    EulerAngles<T> angles;
    // Rotate about z axis until x axis is aligned with y axis
    angles.gamma = M_PI/2;
    
    // Rotate about x axis to get the zenithal angle correct
    angles.beta = M_PI/2-phi;
    
    // Rotate about z axis to get the azimuthal angle correct
    angles.alpha = theta;
    
    // Return
    return angles;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const EulerAngles<T>& angles)
{
    out << "[";
    out << angles.alpha << ",";
    out << angles.beta << ",";
    out << angles.gamma;
    out << "]";
    return out;
}

template <typename T>
bool operator==(const EulerAngles<T>& l, const EulerAngles<T>& r) {
    if (l.alpha != r.alpha) return false;
    if (l.beta != r.beta) return false;
    if (l.gamma != r.gamma) return false;
    
    // All checks passed
    return true;
}

template <typename T>
bool operator!=(const EulerAngles<T>& l, const EulerAngles<T>& r) {
    return !(l==r);
}

template <typename T>
Tensor2<T> EulerZXZRotationMatrix(EulerAngles<T> angle) {
    return EulerZXZRotationMatrix(angle.alpha, angle.beta, angle.gamma);
}

/**
 * @brief Get Euler angles from rotation matrix
 * @param R the rotation matrix
 * @return the angles
 */
template <typename T>
EulerAngles<T> getEulerZXZAngles(Tensor2<T> R)
{      
    EulerAngles<T> angle;
    
    // Angle beta
    angle.beta = std::acos(R(2,2));
    
    if (angle.beta > 1e3*std::numeric_limits<T>::epsilon()) {
        // The other 2 angles
        angle.alpha = std::atan2(R(0,2),-R(1,2));
        angle.gamma = std::atan2(R(2,0),R(2,1));
    }
    else {
        // Singular case
        angle.beta = 0.0;
        T alphaPlusGamma = std::atan2(-R(0,1), R(0,0));
        
        // Not uniquely determined, so just pick a combination
        angle.alpha = alphaPlusGamma/2.0;
        angle.gamma =  alphaPlusGamma/2.0;
    }
    
    // Correct the angle ranges if necessary
    if (angle.alpha < 0) angle.alpha += 2*M_PI;
    if (angle.gamma < 0) angle.gamma += 2*M_PI;
    
    // Return
    return angle;
}

/**
 * @brief Transform tensor \f$ \mathbf{A} \f$ into the frame given by the columns of \f$ \mathbf{Q} \f$
 * @param A \f$ \mathbf{A} \f$
 * @param Q \f$ \mathbf{Q} \f$ 
 * @return \f$ \mathbf{A}^* \f$
 */
template <typename T>
Tensor2<T> transformIntoFrame(const Tensor2<T>& A, const Tensor2<T>& Q) {
    return Q.trans()*A*Q;
}

/**
 * @brief Transform tensor \f$ \mathbf{A}^* \f$ out of the frame given by the columns of \f$ \mathbf{Q} \f$ 
 * @param A_star \f$ \mathbf{A}^* \f$
 * @param Q \f$ \mathbf{Q} \f$ 
 * @return \f$ \mathbf{A} \f$
 */
template <typename T>
Tensor2<T> transformOutOfFrame(const Tensor2<T>& A_star, const Tensor2<T>& Q) {
    return Q*A_star*Q.trans();
}

/**
 * @brief Transform tensor \f$ \mathbf{E}^* \f$ out of the frame given by the columns of \f$ \mathbf{Q} \f$ 
 * @param E_star \f$ \mathbf{E}^* \f$
 * @param Q \f$ \mathbf{Q} \f$ 
 * @return \f$ \mathbf{E} \f$
 * @todo Use symmetries to reduce the computation
 */
template <typename T>
Tensor4<T> transformOutOfFrame(const Tensor4<T>& E_star, const Tensor2<T>& Q) {
    Tensor4<T> E(3,3,3,3);
    for (int m=0; m<3; m++) {
        for (int n=0; n<3; n++) {
            for (int p=0; p<3; p++) {
                for (int q=0; q<3; q++) {
                    T val = 0.0;
                    for (int i=0; i<3; i++) {
                        for (int j=0; j<3; j++) {
                            for (int k=0; k<3; k++) {
                                for (int l=0; l<3; l++) {
                                    val += Q.getVal(m,i)*Q.getVal(n,j)*Q.getVal(p,k)*Q.getVal(q,l)*E_star.getVal(i,j,k,l);
                                }
                            }
                        }
                    }
                    E(m,n,p,q) = val;
                }
            }
        }
    }
    return E;
}

} //END NAMESPACE HPP

#endif /* HPP_TENSOR_H */
