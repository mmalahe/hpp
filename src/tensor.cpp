#include <hpp/tensor.h>
#include <stdio.h>
#include <mpi.h>
#include <cblas.h>
#include <stdlib.h>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <lapacke.h>
#include <type_traits>
#include <algorithm>
#include <exception>
#include <string>

namespace hpp
{

// TENSOR2 //
/////////////

/**
 * @brief Default constructor
 * @details Sets dimensions to zero
 */
template <typename T>
Tensor2<T>::Tensor2() {
    this->n1 = 0;
    this->n2 = 0;
}

// Initialize dimensions
/**
 * @brief Initialize the dimensions and allocation for the tensor
 * @brief The elements are also zeroed.
 * @param n1 the first dimension of the tensor
 * @param n2 the second dimension of the tensor
 * @tparam T the scalar type
 */
template <typename T>
void Tensor2<T>::initialize(const unsigned int n1, const unsigned int n2)
{
    // Dimensions
    this->n1 = n1;
    this->n2 = n2;
    nVals = n1*n2;

    // Allocate
    vals.resize(nVals);
}

/**
 * @brief Constructor for a zeroed tensor
 * @param n1 the first dimension of the tensor
 * @param n2 the second dimension of the tensor
 * @tparam T the scalar type
 * @return the zeroed tensor of dimension n1xn2
 */
template <typename T>
Tensor2<T>::Tensor2(const unsigned int n1, const unsigned int n2)
{
    this->initialize(n1,n2);
}

/**
 * @brief Constructor for a populated tensor
 * @details Populates the values of the tensor from a 1D array of values
 * that have to have the same layout as the the underlying array
 * @param n1 the first dimension of the tensor
 * @param n2 the second dimension of the tensor
 * @param inVals the array to be copied in
 * @tparam T the scalar type
 * @return the populated tensor of dimension n1xn2
 */
template <typename T>
Tensor2<T>::Tensor2(const unsigned int n1, const unsigned int n2, const T* inVals)
{
    this->initialize(n1,n2);
    std::copy(inVals, inVals+nVals, std::begin(vals));
}

/**
 * @brief Constructor for a populated tensor
 * @details Populates the values of the tensor from an STL vector of values
 * that have to have the same layout as the the underlying array
 * @param n1 the first dimension of the tensor
 * @param n2 the second dimension of the tensor
 * @param inVals the STL vector to be copied in
 * @tparam T the scalar type
 * @return the populated tensor of dimension n1xn2
 */
template <typename T>
Tensor2<T>::Tensor2(const unsigned int n1, const unsigned int n2, const std::valarray<T>& inVals)
{
    this->initialize(n1,n2);
    std::copy(std::begin(inVals), std::end(inVals), std::begin(vals));
}

template <typename T>
void Tensor2<T>::constructFromHDF5Dataset(H5::DataSet& dataset, std::vector<hsize_t> gridOffset, std::vector<hsize_t> tensorDims) {
    if (tensorDims.size() != 2) {
        throw TensorError("Incorrect array rank for a rank 2 tensor.");
    }
    this->initialize(tensorDims[0], tensorDims[1]);    
    readSingleHDF5Array(dataset, gridOffset, tensorDims, &(vals[0]));
}

template <typename T>
void Tensor2<T>::constructFromHDF5Dataset(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> gridOffset, std::vector<hsize_t> tensorDims) {
    if (tensorDims.size() != 2) {
        throw TensorError("Incorrect array rank for a rank 2 tensor.");
    }
    this->initialize(tensorDims[0], tensorDims[1]);
    if (HPP_ARRAY_LAYOUT == LAPACK_ROW_MAJOR) {
        readSingleHDF5Array(dset_id, plist_id, gridOffset, tensorDims, &(vals[0]));
    }
    else {
        std::vector<T> readVals(this->getNVals());
        readSingleHDF5Array(dset_id, plist_id, gridOffset, tensorDims, &(readVals[0]));
        for (unsigned int i=0; i<n1; i++) {
            for (unsigned int j=0; j<n2; j++) {
                (*this)(i,j) = readVals[flatC(std::vector<unsigned int>{i,j}, std::vector<unsigned int>{n1,n2})];
            }
        }
    }
}

/**
 * @brief Construct from HD5 dataset and offset
 * @details Using the HDF5 C++ API
 * @param dataset
 * @param gridOffset
 * @param tensorDims
 */
template <typename T>
Tensor2<T>::Tensor2(H5::DataSet& dataset, std::vector<hsize_t> gridOffset, std::vector<hsize_t> tensorDims)
{
    this->constructFromHDF5Dataset(dataset, gridOffset, tensorDims);
}

/**
 * @brief Construct from HD5 file and dataset name
 * @details Where the dataset itself is just the tensor
 * @param file
 * @param datasetName
 */
template <typename T>
Tensor2<T>::Tensor2(H5::H5File& file, const H5std_string& datasetName, std::vector<hsize_t> tensorDims)
{
    //Open dataset
    H5::DataSet dataset = file.openDataSet(datasetName.c_str());
    
    // No actual grid, since there's just the bare dataset containing the tensor
    std::vector<hsize_t> gridOffset; 
    
    // Construct
    this->constructFromHDF5Dataset(dataset, gridOffset, tensorDims);
}

/**
 * @brief Construct from HD5 dataset and offset
 * @details Uses the HDF5 C API
 * @param dset_id
 * @param plist_id
 * @param gridOffset
 * @param tensorDims
 */
template <typename T>
Tensor2<T>::Tensor2(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> gridOffset, std::vector<hsize_t> tensorDims)
{
    this->constructFromHDF5Dataset(dset_id, plist_id, gridOffset, tensorDims);
}

/**
 * @brief Copy constructor
 * @param A the tensor to copy from
 */
template <typename T>
Tensor2<T>::Tensor2(const Tensor2<T>& A)
{
    (*this)=A;
}

/**
 * @brief Construct a square second order tensor from a square fourth order tensor
 * @details This is primarily used for intermediate steps in constructing 
 * fourth order inverses with respect to the contraction between a 4th order
 * tensor and a 2nd order tensor.
 * 
 * The 4th order tensor \f$\mathbf{A}\f$, of dimension \f$n \times n \times n \times n\f$
    is flattened into the 2nd order tensor \f$\mathbf{B}\f$, of dimension \f$ n^2 \times n^2 \f$
    according to:
    \f[
    \mathbf{A}_{ijkl} = \mathbf{B}_{in+j,kn+l}.
    \f]
    That is, it is a row-major flattening.
 * @param A the 4th order tensor \f$\mathbf{A}\f$
 * @tparam T the scalar type
 * @return the 2nd order tensor \f$\mathbf{B}\f$
 */
template <typename T>
Tensor2<T>::Tensor2(const Tensor4<T>& A)
{
    unsigned int dim4 = A.n1;
    if (A.n2 != dim4 || A.n3 != dim4 || A.n4 != dim4) {
        throw TensorError(std::string("Tensor is not square."));
    }
    unsigned int dim2 = dim4*dim4;

    // Initialize
    this->initialize(dim2,dim2);

    // Populate
    for (unsigned int i4=0; i4<dim4; i4++) {
        for (unsigned int j4=0; j4<dim4; j4++) {
            for (unsigned int k4=0; k4<dim4; k4++) {
                for (unsigned int l4=0; l4<dim4; l4++) {
                    unsigned int i2 = i4*dim4+j4;
                    unsigned int j2 = k4*dim4+l4;
                    (*this)(i2,j2) = A.getVal(i4,j4,k4,l4);
                }
            }
        }
    }
}

/**
 * @brief Copy the values from one tensor to another
 * @details Dimension compatability is checked for all builds
 * @param input the input tensor
 * @tparam T the scalar type
 */
template <typename T>
void Tensor2<T>::copyValues(const Tensor2<T>& input)
{
    if (input.n1 != this->n1 || input.n2 != this->n2) {
        throw TensorError(std::string("Size mismatch"));
    }
    for (unsigned int i=0; i<nVals; i++) {
        (*this)(i) = input.getValFlat(i);
    }
}

/**
 * @brief The standard assignment operator
 * @param input the right-hand side of the assignment
 * @tparam T the scalar type
 */
template <typename T>
Tensor2<T>& Tensor2<T>::operator=(const Tensor2<T>& input)
{
    if (!areSameShape(*this, input)) {
        this->initialize(input.n1, input.n2);
    }
    this->copyValues(input);
    return *this;
}

/**
 * @brief Pretty printing to stream
 * @param out
 * @tparam T the scalar type
 */
template <typename T>
void Tensor2<T>::printToStream(std::ostream& out) const
{
    out << "[";
    for (unsigned int i=0; i<n1; i++) {
        out << "[";
        for (unsigned int j=0; j<n2; j++) {
            out << this->getVal(i,j);
            if (j != n2-1) {
                out << ", ";
            }
        }
        out << "]";
        if (i==n1-1) {
            out << "]";
        }
        else {
            out << ",";
            out << std::endl;
        }        
    }
}

template <typename T>
void Tensor2<T>::writeToExistingHDF5Dataset(H5::DataSet& dataset, std::vector<hsize_t> arrayOffset) {
    std::vector<hsize_t> tensorDims = {this->getn1(), this->getn2()};
    writeSingleHDF5Array<T>(dataset, arrayOffset, tensorDims, &(vals[0]));
}

template <typename T>
void Tensor2<T>::writeToExistingHDF5Dataset(hid_t dset_id, hid_t plist_id, std::vector<hsize_t> arrayOffset) {
    std::vector<hsize_t> tensorDims = {this->getn1(), this->getn2()};
    writeSingleHDF5Array<T>(dset_id, plist_id, arrayOffset, tensorDims, &(vals[0]));
}

/**
 * @brief Adding to an HDF5 file as a new dataset
 * @param file the HDF5 file handle
 * @param datasetName the name of this dataset
 */
template <typename T>
void Tensor2<T>::addAsNewHDF5Dataset(H5::H5File& file, const H5std_string& datasetName) {
    // Dimensions of HDF5 array
    hsize_t dims[2];
    dims[0] = n1;
    dims[1] = n2;
    const int tensor_rank = 2;
    H5::DataSpace dataspace(tensor_rank, dims);
    
    // HDF5 type
    H5::DataType dataType = getHDF5Type<T>();
    
    // Create empty dataset
    H5::DataSet dataset = file.createDataSet(datasetName.c_str(), dataType, dataspace);
    
    // Write to dataset
    dataset.write(&(vals[0]), dataType);    
}

template <typename T>
void Tensor2<T>::writeToNewHDF5(const H5std_string& filename, const H5std_string& datasetName) {
    // New file, replacing any existing one
    H5::H5File file(filename.c_str(), H5F_ACC_TRUNC);
    
    // Add dataset
    this->addAsNewHDF5Dataset(file, datasetName);
}

// Assert squareness
template <typename T>
void Tensor2<T>::assertSquare() const{
    if (n1 != n2) {
        throw TensorError(std::string("Tensor is not square."));
    }
}

// Check if rotation matrix
template <typename T>
bool Tensor2<T>::isRotationMatrix() const{
    this->assertSquare();
    Tensor2<T> product = (*this)*(*this).trans();
    T error = (product - identityTensor2<T>(n1)).frobeniusNorm();
    if (error < 1000*std::numeric_limits<T>::epsilon()) {
        return true;
    }
    else {
        return false;
    }
}


/**
 * @brief The inverse of the tensor
 * @return the inverse of this tensor, \f$\mathbf{A}^{-1}\f$
 * @tparam T the scalar type
 */
template <typename T>
void Tensor2<T>::invInPlace()
{
    this->assertSquare();
    
    // LU factorize //
    //////////////////
    
    // Layout
    int matrix_layout = HPP_ARRAY_LAYOUT;
    
    // Matrix dimensions
    lapack_int m = this->n1;
    lapack_int n = this->n2;
    lapack_int lda = std::max(m,n);
    
    // Array of pivots
    int min_dim = std::min(m,n);
    std::vector<lapack_int> ipiv(min_dim);
    
    // Returns
    lapack_int info;
    
    // LU factorize with LAPACKE getrf
    if (std::is_same<T, double>::value) {
        info = LAPACKE_dgetrf(matrix_layout, m, n, (double*)&(vals[0]), lda, ipiv.data());
    } else if (std::is_same<T, float>::value) {
       info = LAPACKE_sgetrf(matrix_layout, m, n, (float*)&(vals[0]), lda, ipiv.data()); 
    }
    // a now contains the P*L*U factorization
    
    // Check result
    if (info != 0) {
        if (info < 0) {
            DEBUG_ONLY(std::cerr << "Argument " << -info << " is illegal." << std::endl;)
            throw TensorError(std::string("LAPACKE getrf call failed."));
        } else if (info > 0) {
            DEBUG_ONLY(std::cerr << "Matrix is singular." << std::endl);
            DEBUG_ONLY(std::cerr << "Matrix is: " << *this << std::endl);
            throw TensorError(std::string("LAPACKE getrf call failed."));
        }
    }
    
    // INVERT //
    ////////////
    
    // Invert with LAPACKE_dgetri
    if (std::is_same<T, double>::value) {
        info = LAPACKE_dgetri(matrix_layout, m, (double*)&(vals[0]), lda, ipiv.data());
    } else if (std::is_same<T, float>::value) {
        info = LAPACKE_sgetri(matrix_layout, m, (float*)&(vals[0]), lda, ipiv.data());
    }
    // a now contains the inverted matrix
    
    // Check result
    if (info != 0) {
        if (info < 0) {
            DEBUG_ONLY(std::cerr << "Argument " << -info << " is illegal." << std::endl;)
            throw TensorError(std::string("LAPACKE getri call failed."));
        } else if (info > 0) {
            DEBUG_ONLY(std::cerr << "Matrix is singular." << std::endl;)
            throw TensorError(std::string("LAPACKE getri call failed."));
        }
    }
}

/**
 * @brief The inverse of the tensor
 * @return the inverse of this tensor, \f$ \mathbf{A}^{-1} \f$
 * @tparam T the scalar type
 */
template <typename T>
Tensor2<T> Tensor2<T>::inv() const
{    
    // Create new tensor for inverse
    Tensor2<T> AInv = *this;
    
    // Invert
    AInv.invInPlace();
    
    // Return
    return AInv;
}



/**
 * @brief The trace of the tensor
 * @tparam T the scalar type
 * @return the trace of this tensor, \f$tr(\mathbf{A})\f$
 */
template <typename T>
T Tensor2<T>::tr() const{
    this->assertSquare();
    T trace = 0;
    for (unsigned int i=0; i<n1; i++) {
        trace += this->getVal(i,i);
    }
    return trace;
}

/**
 * @brief The determinant of the tensor
 * @details The approach is taken from 
 * https://www.ibm.com/developerworks/community/blogs/jfp/entry/A_Comparison_Of_C_Julia_Python_Numba_Cython_Scipy_and_BLAS_on_LU_Factorization
 * It is not particularly stable, and we should consider a more robust 
 * implementation for large tensors
 * @tparam T the scalar type
 * @return the determinant of this tensor, \f$det(\mathbf{A})\f$
 */
template <typename T>
T Tensor2<T>::det() const {
    this->assertSquare();
    
    // Layout
    int matrix_layout = HPP_ARRAY_LAYOUT;
    
    // Matrix dimensions
    lapack_int m = this->n1;
    lapack_int n = this->n2;
    lapack_int lda = std::max(m,n);
    
    // Array into and out of GETRF
    std::valarray<T> a = vals;
    
    // Array of pivots
    int min_dim = std::min(m,n);
    std::vector<lapack_int> ipiv(min_dim);
    
    // Returns
    lapack_int info;
    
    // LU factorize with LAPACKE getrf
    if (std::is_same<T, double>::value) {
       info = LAPACKE_dgetrf(matrix_layout, m, n, (double*)&(a[0]), lda, ipiv.data());
    } else if (std::is_same<T, float>::value) {
       info = LAPACKE_sgetrf(matrix_layout, m, n, (float*)&(a[0]), lda, ipiv.data()); 
    }
    // a now contains the P*L*U factorization
    
    // Check result
    if (info != 0) {
        if (info < 0) {
            DEBUG_ONLY(std::cerr << "Argument " << -info << " is illegal." << std::endl;)
            throw TensorError(std::string("LAPACKE getrf call failed."));
        } else if (info > 0) {
            DEBUG_ONLY(std::cerr << "Matrix is singular." << std::endl;)
            DEBUG_ONLY(std::cerr << "Matrix is: " << *this << std::endl;)
            throw TensorError(std::string("LAPACKE getrf call failed."));
        }
    }
    
    // Magnitude of determinant is now product of U diagonals
    T determinant = 1.0;
    for (int i=0; i<m; i++) {
        determinant *= a[this->flat(i,i)];
    }
    
    // Sign of determinant is determined by number of row interchanges (each have determinant -1)
    for (int i=0; i<m; i++) {
        // The i+1 is necessary because the pivot indices start at 1 (LAPACK's FORTRAN legacy)
        if (ipiv[i] != i+1) {
            determinant *= -1.0;
        }
    }
    
    // Return
    return determinant;    
}

/**
 * @brief The transpose of the tensor
 * @tparam T the scalar type
 * @return the tranpose of this tensor, \f$\mathbf{A}^T\f$
 */
template <typename T>
Tensor2<T> Tensor2<T>::trans() const{
    Tensor2<T> A(n2,n1);
    for (unsigned int i=0; i<n1; i++) {
        for (unsigned int j=0; j<n2; j++) {
            A(j,i) = this->getVal(i,j);
        }
    }
    return A;
}

/**
 * @brief The element-wise absolute value of the tensor
 * @tparam T the scalar type
 * @return the element-wise absolute value of the tensor
 */
template <typename T>
Tensor2<T> Tensor2<T>::abs() const {
    Tensor2<T> A(n1, n2, std::abs(vals));
    return A;
}

/**
 * @brief The minimum element in the tensor
 * @tparam T the scalar type
 * @return the minimum element in the tensor
 */
template <typename T>
T Tensor2<T>::min() const {
    return vals.min();
}

/**
 * @brief The maximum element in the tensor
 * @tparam T the scalar type
 * @return the maximum element in the tensor
 */
template <typename T>
T Tensor2<T>::max() const {
    return vals.max();
}

/**
 * @brief The maximum element in the tensor
 * @tparam T the scalar type
 * @return the maximum element in the tensor
 */
template <typename T>
T Tensor2<T>::absmax() const {
    return std::abs(vals).max();
}

/**
 * @brief Product of all elements
 * @tparam T the scalar type
 * @return The prodct
 */
template <typename T>
T Tensor2<T>::prod() const {
    T product;
    for (unsigned int i=0; i<nVals; i++) {
        product *= vals[i];
    }
    return product;
}

/**
 * @brief Creates a new tensor with the values of the elements constrained
 * between two values
 * @details For each element, if its value is below minVal, it's set to minVal.
 * If it's above maxVal, it's set to maxVal.
 * @param minVal the minimum value
 * @param maxVal the maximum value
 * @return a new tensor that conforms to the constraints
 */
template <typename T>
Tensor2<T> Tensor2<T>::constrainedTo(const T minVal, const T maxVal) const {
    if (minVal > maxVal) {
        throw TensorError(std::string("Min is greater than max."));
    }
    std::valarray<T> constrainedVec = vals;
    for (auto&& val : constrainedVec) {
        if (val < minVal) {
            val = minVal;
        }
        else if (val > maxVal) {
            val = maxVal;
        }
    }
    Tensor2<T> A(n1, n2, constrainedVec);
    return A;
}

/**
 * @brief Replaces the elements with their values constrained
 * between two limits
 * @details For each element, if its value is below minVal, it's set to minVal.
 * If it's above maxVal, it's set to maxVal.
 * @param minVal the minimum value
 * @param maxVal the maximum value
 */
template <typename T>
void Tensor2<T>::constrainInPlace(const T minVal, const T maxVal){
    if (minVal > maxVal) {
        throw TensorError(std::string("Min is greater than max."));
    }
    for (auto&& val : vals) {
        if (val < minVal) {
            val = minVal;
        }
        else if (val > maxVal) {
            val = maxVal;
        }
    }
}

/**
 * @brief Create a tensor with the values uniformly scaled so that
 * the new tensor has unit determinant.
 * @tparam T the scalar type
 * @return the scaled tensor
 */
template <typename T>
Tensor2<T> Tensor2<T>::scaledToUnitDeterminant() const {
    T det = this->det();
    if (det < 0) {
        throw TensorError(std::string("Determinant scaling is only for tensors with positive determinants."));
    }
    T detNthRoot = std::pow(det,1.0/(this->n1));
    return (*this)/detNthRoot;
}

/**
 * @brief The deviatoric component of the tensor
 * @tparam T the scalar type
 * @return the deviatoric component of the tensor
 */
template <typename T>
Tensor2<T> Tensor2<T>::deviatoricComponent() const {
    Tensor2<T> A = (*this);
    T meanNormalComponent = this->tr()/n1;
    for (unsigned int i=0; i<n1; i++) {
        A(i,i) -= meanNormalComponent;
    }
    return A;
}

template <typename T>
void Tensor2<T>::evecDecomposition(std::valarray<T>& evals, Tensor2<T>& evecs) {    
    // Layout
    int matrix_layout = HPP_ARRAY_LAYOUT;
    
    // Job
    const char jobz = 'V'; // compute eigenvalues and eigenvectors
    const char range = 'A'; // compute all of them
    const char uplo = 'U'; // upper triangle of A is stored
    
    // Matrix dimensions
    lapack_int m = this->n1;
    lapack_int n = this->n2;
    lapack_int lda = std::max(m,n);
    
    // Array into and out of GETRF
    std::valarray<T> a = vals;
    
    // Search ranges. 
    // Not referenced, since we want all eigenvalues
    T vl = 0;
    T vu = 0;
    lapack_int il = 0;
    lapack_int iu = 0;
    
    // Convergence
    // Negative abstol results in using default value
    T abstol = -1.0; 
    
    // Eigenvalues
    lapack_int nEvalsFound;
    evals = std::valarray<T>(n);
    
    // Eigenvectors
    evecs = Tensor2<T>(n,n);
    lapack_int ldz = std::max(1,n);
    
    // Support of the eigenvectors
    std::vector<lapack_int> isuppz(2*n);
    
    // Returns
    // Initial value is arbitrary and only set to suppress warnings about a lack
    // of initialisation.
    lapack_int info = 1;
    
    // LU factorize with LAPACKE getrf
    if (std::is_same<T, double>::value) {
       info = LAPACKE_dsyevr(matrix_layout, jobz, range, uplo, n, (double*)&(a[0]),
                             lda, vl, vu, il, iu, abstol, &nEvalsFound, (double*)&(evals[0]),
                             (double*)&(evecs.vals[0]), ldz, isuppz.data());
                             
    } else if (std::is_same<T, float>::value) {
       info = LAPACKE_ssyevr(matrix_layout, jobz, range, uplo, n, (float*)&(a[0]),
                             lda, vl, vu, il, iu, abstol, &nEvalsFound, (float*)&(evals[0]),
                             (float*)&(evecs.vals[0]), ldz, isuppz.data());
    }
    
    // Check result
    if (info != 0) {
        if (info < 0) {
            DEBUG_ONLY(std::cerr << "Argument " << -info << " is illegal." << std::endl;)
            throw TensorError("LAPACKE syevr call failed.");
        } else if (info > 0) {
            DEBUG_ONLY(std::cerr << "Internal error." << std::endl;)
            throw TensorError("LAPACKE syevr call failed.");
        }
    }
}

template <typename T>
T Tensor2<T>::frobeniusNorm() const{
    // The norm
    T norm = 0.0;    
    for (unsigned int i=0; i<vals.size(); i++) {
        norm += std::pow(std::abs(vals[i]), 2.0);
    }
    norm = std::sqrt(norm);
    
    // Return
    return norm;
}

/**
 * @brief Matrix square root of the tensor
 * @return 
 */
template <typename T>
Tensor2<T> Tensor2<T>::sqrtMatrix() {    
    // Must be square
    this->assertSquare();
    
    // Eigenvalues
    std::valarray<T> evals;
    
    // Eigenvectors
    Tensor2<T> evecs;    
    
    // Get decomposition
    this->evecDecomposition(evals, evecs);
    
    // Check eigenvalues are all positive
    for (unsigned int i=0; i<this->n1; i++) {
        if (evals[i] < 0.0) {
            throw TensorError("Matrix is not positive semi-definite.");
        }
    }
    
    // Construct the square root
    std::valarray<T> evalsSqrt = std::sqrt(evals);
    Tensor2<T> A = (evecs*diag2(evalsSqrt))*(evecs.inv());
    
    // Return
    return A;
}

/**
 * @brief Polar decomposition of the tensor
 * @return 
 */
template <typename T>
PolarDecomposition<T> Tensor2<T>::polarDecomposition() const{
    PolarDecomposition<T> decomp;
    decomp.U = (this->trans()*(*this)).sqrtMatrix();
    decomp.R = (*this)*(decomp.U.inv());
    return decomp;
}

// TENSOR4 //
/////////////
/**
 * @brief Default constructor
 * @details Sets dimensions to zero
 */
template <typename T>
Tensor4<T>::Tensor4() {
    this->n1 = 0;
    this->n2 = 0;
    this->n3 = 0;
    this->n4 = 0;
}

/**
 * @brief Constructor for Tensor2D
 * It is responsible for setting the dimensions of the tensor, including
 * the total number of elements. It is also responsible for allocating memory
 * for the underlying array, and zeroing it.
 * @param n1 the first dimension of the tensor
 * @param n2 the second dimension of the tensor
 * @param n3 the third dimension of the tensor
 * @param n4 the fourth dimension of the tensor
 */
template <typename T>
Tensor4<T>::Tensor4(unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4)
{
    this->initialize(n1, n2, n3, n4);
}

template <typename T>
void Tensor4<T>::initialize(unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4)
{
    // Dimensions
    this->n1 = n1;
    this->n2 = n2;
    this->n3 = n3;
    this->n4 = n4;
    nVals = n1*n2*n3*n4;

    // Allocate
    vals.resize(nVals);
}

// Copy constructor
template <typename T>
Tensor4<T>::Tensor4(const Tensor4<T>& A)
{
    this->initialize(A.n1,A.n2,A.n3,A.n4);
    (*this)=A;
}

// Copying values
template <typename T>
void Tensor4<T>::copyValues(const Tensor4<T>& input)
{
    if (input.n1 != this->n1 || input.n2 != this->n2 || input.n3 != this->n3 || input.n4 != this->n4) {
        throw TensorError(std::string("Size mismatch"));
    }
    for (unsigned int i=0; i<nVals; i++) {
        (*this)(i) = input.getValFlat(i);
    }
}

// Assignment operator
template <typename T>
Tensor4<T>& Tensor4<T>::operator=(const Tensor4<T>& input)
{
    if (!areSameShape(*this, input)) {
        this->initialize(input.n1, input.n2, input.n3, input.n4);
    }
    this->copyValues(input);
    return *this;
}

template <typename T>
Tensor4<T>::Tensor4(const Tensor2<T>& A)
{
    unsigned int dim2 = A.n1;
    if (A.n2 != dim2) throw TensorError(std::string("Tensor is not square."));
    float dim4Float = sqrt((float)dim2);
    unsigned int dim4 = (unsigned int) dim4Float;
    if (dim4 != dim4Float) throw TensorError(std::string("Resulting tensor will not be square."));

    // Initialize
    this->initialize(dim4,dim4,dim4,dim4);

    // Populate
    for (unsigned int i4=0; i4<dim4; i4++) {
        for (unsigned int j4=0; j4<dim4; j4++) {
            for (unsigned int k4=0; k4<dim4; k4++) {
                for (unsigned int l4=0; l4<dim4; l4++) {
                    unsigned int i2 = i4*dim4+j4;
                    unsigned int j2 = k4*dim4+l4;
                    (*this)(i4,j4,k4,l4) = A.getVal(i2,j2);
                }
            }
        }
    }
}

/**
 * @brief Destructor for Tensor2D
 * It is responsible for freeing and invalidating the underlying array.
 */
template <typename T>
Tensor4<T>::~Tensor4()
{
}

template <typename T>
void Tensor4<T>::printToStream(std::ostream& out)
{
    out << "[";
    for (unsigned int i=0; i<n1; i++) {
        if (i != 0) {
            out << " ";
        }
        out << "{";
        for (unsigned int j=0; j<n2; j++) {
            if (j != 0) {
                out << " ";
            }
            out << "[";
            for (unsigned int k=0; k<n3; k++) {
                if (k != 0) {
                    out << " ";
                }
                out << "[";
                for (unsigned int l=0; l<n4; l++) {
                    if (l != 0) {
                        out << " ";
                    }
                    out << (*this)(i,j,k,l);
                }
                out << "]";
                if (k==n3-1) {
                    out << "]";
                }
                out << std::endl;
            }
            if (j==n2-1) {
                out << "]";
            }
        }
        if (i==n1-1) {
            out << "]";
        }
    }
    out << std::endl;
}

// Should probably assert that it's square here
template <typename T>
void Tensor4<T>::invInPlace() {
    Tensor2<T> A2(*this);
    A2.invInPlace();
    identityTensor4InPlace(this->n1, *this);
    Tensor2<T> I2(*this);
    Tensor2<T> A2I2 = A2*I2;
    *this = Tensor4<T>(A2I2);
}

// Should probably assert that it's square here
template <typename T>
Tensor4<T> Tensor4<T>::inv() const {
    Tensor4<T> A = *this;
    A.invInPlace();
    return A;
}

template <typename T>
T Tensor4<T>::frobeniusNorm() const{
    // The norm
    T norm = 0.0;    
    for (unsigned int i=0; i<vals.size(); i++) {
        norm += std::pow(std::abs(vals[i]), 2.0);
    }
    norm = std::sqrt(norm);
    
    // Return
    return norm;
}

// Tensor2 and Tensor4 are restricted to these specific instantiations
template class Tensor2<float>;
template class Tensor2<double>;
template class Tensor4<float>;
template class Tensor4<double>;

} //END NAMESPACE HPP