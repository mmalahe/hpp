/** @file tensorCUDA.h
* @author Michael Malahe
* @brief Header file for tensor classes CUDA implementations.
* @details Note that for all of these implementations, dynamic memory is not used.
* All of the memory lives on whichever architecture the class is instantiated on.
* That is, the "CUDA" suffix indicates nothing about where the memory is, but
* just indicates that it's in a format that's most suitable for a CUDA implementation.
*/

#ifndef TENSOR_CUDA_H
#define TENSOR_CUDA_H

#include <initializer_list>
#include <hpp/config.h>
#include <hpp/tensor.h>
#include <hpp/cudaUtils.h>
#include <hpp/hdfUtilsCpp.h>

namespace hpp
{
#ifdef HPP_USE_CUDA

// Forward declarations
template <typename U, unsigned int N>
class VecCUDA;
template <typename T, unsigned int M, unsigned int N>
class Tensor2CUDA;
template <typename T, unsigned int N>
class Tensor2SymmCUDA;
template <typename T, unsigned int N>
class Tensor2AsymmCUDA;


/////////////
// VECCUDA //
/////////////
template <typename T, unsigned int N>
class VecCUDA {
public:
     __host__ __device__ VecCUDA(){
         for (unsigned int i=0; i<N; i++) {
             vals[i] = (T)0.0;
         }
    }
    
    // Constructors
    VecCUDA(const std::vector<T>& in);
    VecCUDA(const std::initializer_list<T>& in);
    
    // Iterators
    T* begin() {return &(vals[0]);}
    T* end() {return begin()+N;}
    const T* begin() const {return &(vals[0]);}
    const T* end() const {return begin()+N;}
    
    // Get read/write reference to value
    __host__ __device__ T& operator()(const unsigned int i) {
        return vals[i];
    }
    // Get read-only value
    __host__ __device__ T getVal(const unsigned int i) const {
        return vals[i];
    }
    // Set value
    __host__ __device__ void setVal(const unsigned int i, const T val) {
        vals[i] = val;
    }
    
    // Norm
    __device__ T norm() const{
        T sumOfSquares = (T)0.0;
        for (unsigned int i=0; i<N; i++) {
            sumOfSquares += vals[i]*vals[i];
        }
        return sqrtIntrinsic(sumOfSquares);
    }
protected:
    T vals[N];
};

// Constructors
template <typename T, unsigned int N>
VecCUDA<T,N>::VecCUDA(const std::vector<T>& in) {
    // Check size
    if (N != in.size()) {
        throw TensorError("Size mismatch.");
    }
    
    // Copy
    std::copy(in.begin(), in.end(), &(vals[0]));
}

template <typename T, unsigned int N>
VecCUDA<T,N>::VecCUDA(const std::initializer_list<T>& in) {
    // Check size
    if (N != in.size()) {
        throw TensorError("Size mismatch.");
    }
    
    // Copy
    std::copy(in.begin(), in.end(), &(vals[0]));
}

// Operators
template <typename T, unsigned int N>
__device__ VecCUDA<T,N> operator/(const VecCUDA<T,N>& inVec, T scalar) {
    VecCUDA<T,N> outVec;
    for (unsigned int i=0; i<N; i++) {
        outVec(i) = inVec.getVal(i)/scalar;
    }
    return outVec;
}

/**
 * @brief Uses the "mathematics" convention.
 * @param cartVec the vector in cartesian coordinates
 * @return the vector in spherical coordinates
 */
template <typename T>
__device__ VecCUDA<T,3> cartesianToSpherical(const VecCUDA<T,3>& cartVec) {
    // Magnitude
    T r = cartVec.norm();
    VecCUDA<T,3> unitVec = cartVec/r;
    
    // Azimuthal component
    T theta = atan2(unitVec(1), unitVec(0));
    
    // Polar
    T phi = acos(unitVec(2));    
    
    // Return
    VecCUDA<T,3> sphereVec;
    sphereVec(0) = r;
    sphereVec(1) = theta;
    sphereVec(2) = phi;
    return sphereVec;
}

//////////////
// TENSOR 2 //
//////////////

template <typename T, unsigned int M, unsigned int N>
class Tensor2CUDA {
public:
    // Default constructor
    __host__ __device__ Tensor2CUDA(){
        for (unsigned int i=0; i<M; i++) {
            for (unsigned int j=0; j<N; j++) {
                vals[i][j] = (T)0.0;
            }
        }
    }
    
    // Construct from standard Tensor2
    Tensor2CUDA(const Tensor2<T>& in);
    
    // Construct from symmetric CUDA tensor
    __host__ __device__ Tensor2CUDA(const Tensor2SymmCUDA<T,N>& input) {
        for (unsigned int i=0; i<N; i++) {
            for (unsigned int j=0; j<N; j++) {
                vals[i][j] = input.getVal(i,j); 
            }
        }
    }
    
    // Assignment
    __host__ __device__ Tensor2CUDA<T,M,N>& operator=(const Tensor2CUDA<T,M,N>& input) {
        memcpy(vals, input.vals, M*N*sizeof(T));
        return *this;
    }
    
    // Assign from symmetric CUDA tensor
    __host__ __device__ Tensor2CUDA<T,N,N>& operator=(const Tensor2SymmCUDA<T,N>& input) {
        for (unsigned int i=0; i<N; i++) {
            for (unsigned int j=0; j<N; j++) {
                vals[i][j] = input.getVal(i,j); 
            }
        }
        return *this;
    }    
    
    // Get read/write reference to value
    __host__ __device__ T& operator()(const unsigned int i, const unsigned int j) {
        return vals[i][j];
    }
    // Get read-only value
    __host__ __device__ T getVal(const unsigned int i, const unsigned int j) const {
        return vals[i][j];
    }
    // Set value
    __host__ __device__ void setVal(const unsigned int i, const unsigned int j, const T val) {
        vals[i][j] = val;
    }
    
    // Transpose
    __host__ __device__ Tensor2CUDA<T,N,M> trans() const {
        Tensor2CUDA<T,N,M> A;
        for (unsigned int i=0; i<M; i++) {
            for (unsigned int j=0; j<N; j++) {
                A(j,i) = this->getVal(i,j);
            }
        }
        return A;
    }
    
    /**
     * @brief 3x3 inverse
     * @details No attempts at singularity checking or significant optimisations.
     */
    __host__ __device__ Tensor2CUDA<T,3,3> inv() const {
        // Determinant
        T det = this->getVal(0,0)*(this->getVal(1,1)*this->getVal(2,2) - this->getVal(2,1)*this->getVal(1,2));
        det -=  this->getVal(0,1)*(this->getVal(1,0)*this->getVal(2,2) - this->getVal(1,2)*this->getVal(2,0));
        det +=  this->getVal(0,2)*(this->getVal(1,0)*this->getVal(2,1) - this->getVal(1,1)*this->getVal(2,0));
        T ooDet = (T)1.0/det;
        
        // Inverse
        Tensor2CUDA<T,3,3> A;
        A(0,0) = (this->getVal(1,1)*this->getVal(2,2) - this->getVal(2,1)*this->getVal(1,2))*ooDet;
        A(0,1) = (this->getVal(0,2)*this->getVal(2,1) - this->getVal(0,1)*this->getVal(2,2))*ooDet;
        A(0,2) = (this->getVal(0,1)*this->getVal(1,2) - this->getVal(0,2)*this->getVal(1,1))*ooDet;
        A(1,0) = (this->getVal(1,2)*this->getVal(2,0) - this->getVal(1,0)*this->getVal(2,2))*ooDet;
        A(1,1) = (this->getVal(0,0)*this->getVal(2,2) - this->getVal(0,2)*this->getVal(2,0))*ooDet;
        A(1,2) = (this->getVal(1,0)*this->getVal(0,2) - this->getVal(0,0)*this->getVal(1,2))*ooDet;
        A(2,0) = (this->getVal(1,0)*this->getVal(2,1) - this->getVal(2,0)*this->getVal(1,1))*ooDet;
        A(2,1) = (this->getVal(2,0)*this->getVal(0,1) - this->getVal(0,0)*this->getVal(2,1))*ooDet;
        A(2,2) = (this->getVal(0,0)*this->getVal(1,1) - this->getVal(1,0)*this->getVal(0,1))*ooDet;
        
        // Return
        return A;
    }
    
    // Print to stream
    void printToStream(std::ostream& out) const;
    
    // Write to HDF5
    void writeToExistingHDF5Dataset(H5::DataSet& dataset, std::vector<hsize_t> arrayOffset) {
        std::vector<hsize_t> tensorDims = {M, N};
        writeSingleHDF5Array<T>(dataset, arrayOffset, tensorDims, &(vals[0][0]));
    }
    
    // Friends
private:
    // Tensor values
    T vals[M][N];
};

// Constructors
template <typename T, unsigned int M, unsigned int N>
Tensor2CUDA<T,M,N>::Tensor2CUDA(const Tensor2<T>& in) {
    // Check size
    if (M != in.n1 || N != in.n2) {
        throw TensorError("Size mismatch.");
    }
    
    // Copy
    std::copy(&(in.vals[0]), &(in.vals[0])+M*N, &(vals[0][0]));    
}

// Subtraction
template <typename T, unsigned int M, unsigned N>
__host__ __device__ Tensor2CUDA<T,M,N> operator-(const Tensor2CUDA<T,M,N>& A, const Tensor2CUDA<T,M,N>& B) {
    Tensor2CUDA<T,M,N> C;
    for (unsigned int i=0; i<M; i++) {
        for (unsigned int j=0; j<N; j++) {
            C(i,j) = A.getVal(i,j) - B.getVal(i,j);
        }
    }
    return C;
}

// Addition
template <typename T, unsigned int M, unsigned N>
__host__ __device__ Tensor2CUDA<T,M,N> operator+(const Tensor2CUDA<T,M,N>& A, const Tensor2CUDA<T,M,N>& B) {
    Tensor2CUDA<T,M,N> C;
    for (unsigned int i=0; i<M; i++) {
        for (unsigned int j=0; j<N; j++) {
            C(i,j) = A.getVal(i,j) + B.getVal(i,j);
        }
    }
    return C;
}
template <typename T, unsigned int M, unsigned N>
__host__ __device__ void operator+=(Tensor2CUDA<T,M,N>& A, const Tensor2CUDA<T,M,N>& B) {
    A = A+B;
}

// Scalar Multiplication
template <typename T, unsigned int M, unsigned N>
__host__ __device__ Tensor2CUDA<T,M,N> operator*(const Tensor2CUDA<T,M,N>& A, T scalar) {
    Tensor2CUDA<T,M,N> B;
    for (unsigned int i=0; i<M; i++) {
        for (unsigned int j=0; j<N; j++) {
            B(i,j) = scalar*A.getVal(i,j);
        }
    }
    return B;
}
template <typename T, unsigned int M, unsigned N>
__host__ __device__ Tensor2CUDA<T,M,N> operator*(T scalar, const Tensor2CUDA<T,M,N>& A) {
    return A*scalar;
}

// Scalar division
template <typename T, unsigned int M, unsigned N>
__host__ __device__ Tensor2CUDA<T,M,N> operator/(const Tensor2CUDA<T,M,N>& A, T scalar) {
    Tensor2CUDA<T,M,N> B;
    for (unsigned int i=0; i<M; i++) {
        for (unsigned int j=0; j<N; j++) {
            B(i,j) = A.getVal(i,j)/scalar;
        }
    }
    return B;
}
template <typename T, unsigned int M, unsigned N>
__host__ __device__ void operator/=(Tensor2CUDA<T,M,N>& A, T scalar) {
    A = A/scalar;}


// Matrix multiplication
template <typename T, unsigned int M, unsigned int N, unsigned int P>
__host__ __device__ Tensor2CUDA<T,M,P> operator*(const Tensor2CUDA<T,M,N>& A, const Tensor2CUDA<T,N,P>& B) {
    Tensor2CUDA<T,M,P> C;
    for (unsigned int i=0; i<M; i++) {
        for (unsigned int j=0; j<P; j++) {
            for (unsigned int k=0; k<N; k++) {
                C(i,j) += A.getVal(i,k)*B.getVal(k,j);
            }
        }
    }
    return C;
}

// Matrix-vector multiplication
template <typename T, unsigned int M, unsigned int N>
__host__ __device__ VecCUDA<T,M> operator*(const Tensor2CUDA<T,M,N>& A, const VecCUDA<T,N>& x) {
    VecCUDA<T,M> b;
    for (unsigned int i=0; i<M; i++) {
        for (unsigned int j=0; j<N; j++) {
            b(i) += A.getVal(i,j)*x.getVal(j);
        }
    }
    return b;
}

/**
 * @brief Transform tensor \f$ \mathbf{A} \f$ into the frame given by the columns of \f$ \mathbf{Q} \f$
 * @param A \f$ \mathbf{A} \f$
 * @param Q \f$ \mathbf{Q} \f$ 
 * @return \f$ \mathbf{A}^* \f$
 */
template <typename T, unsigned int M>
__host__ __device__ Tensor2CUDA<T,M,M> transformIntoFrame(const Tensor2CUDA<T,M,M>& A, const Tensor2CUDA<T,M,M>& Q) {
    return Q.trans()*A*Q;
}

/**
 * @brief Transform tensor \f$ \mathbf{A}^* \f$ out of the frame given by the columns of \f$ \mathbf{Q} \f$ 
 * @param A_star \f$ \mathbf{A}^* \f$
 * @param Q \f$ \mathbf{Q} \f$ 
 * @return \f$ \mathbf{A} \f$
 */
template <typename T, unsigned int M>
__host__ __device__ Tensor2CUDA<T,M,M> transformOutOfFrame(const Tensor2CUDA<T,M,M>& A_star, const Tensor2CUDA<T,M,M>& Q) {
    return Q*A_star*Q.trans();
}

template <typename T, unsigned int M>
__host__ __device__ Tensor2CUDA<T,M,M> transformIntoFrame(const Tensor2AsymmCUDA<T,M>& A, const Tensor2CUDA<T,M,M>& Q) {
    return Q.trans()*A*Q;
}

template <typename T, unsigned int M>
__host__ __device__ Tensor2CUDA<T,M,M> transformOutOfFrame(const Tensor2AsymmCUDA<T,M>& A_star, const Tensor2CUDA<T,M,M>& Q) {
    return Q*A_star*Q.trans();
}

template <typename T>
__device__ Tensor2CUDA<T,3,3> EulerZXZRotationMatrixCUDA(T alpha, T beta, T gamma) {
    Tensor2CUDA<T,3,3> R;
    T c1, c2, c3, s1, s2, s3;
    sincosIntrinsic(alpha, &s1, &c1);
    sincosIntrinsic(beta, &s2, &c2);
    sincosIntrinsic(gamma, &s3, &c3);
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

// Printing to stream
template <typename T, unsigned int M, unsigned int N>
void Tensor2CUDA<T,M,N>::printToStream(std::ostream& out) const
{
    out << "[";
    for (unsigned int i=0; i<M; i++) {
        out << "[";
        for (unsigned int j=0; j<N; j++) {
            out << this->getVal(i,j);
            if (j != N-1) {
                out << ", ";
            }
        }
        out << "]";
        if (i==M-1) {
            out << "]";
        }
        else {
            out << ",";
        }
        out << std::endl;
    }
}

// Stream output
template <typename T, unsigned int M, unsigned int N>
std::ostream& operator<<(std::ostream& out, const Tensor2CUDA<T,M,N>& A)
{
    A.printToStream(out);
    return out;
}

// PARALLEL REDUCTION //
/**
 * @brief 
 * @details Note that this is valid only on CC 3.0 and above. 
 * Adapted from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 * @param val
 * @return 
 */
template <typename T, unsigned int M, unsigned N>
inline __device__ Tensor2CUDA<T,M,N> warpReduceSumTensor2(Tensor2CUDA<T,M,N> A) {
    const int warpSize = 32;
    for (unsigned int i=0; i<M; i++) {
        for (unsigned int j=0; j<N; j++) {
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                A(i,j) += __shfl_down(A(i,j), offset);
            }
        }
    }
    return A;
}

/**
 * @brief
 * @details Adapted from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 * @param val
 */
template <typename T, unsigned int M, unsigned N>
inline __device__ Tensor2CUDA<T,M,N> blockReduceSumTensor2(Tensor2CUDA<T,M,N> val) {
    const int warpSize = 32;
    static __shared__ Tensor2CUDA<T,M,N> shared[warpSize]; // Shared mem for 32 partial sums
    __syncthreads();
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSumTensor2(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    if (threadIdx.x < blockDim.x / warpSize) {
        val = shared[lane];
    }
    else {
        val = Tensor2CUDA<T,M,N>();
    }

    if (wid==0) val = warpReduceSumTensor2(val); //Final reduce within first warp

    return val;
}

/**
 * @brief 
 * @details Adapted from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 * @param in
 * @param out
 * @param N
 */
template <typename T, unsigned int M, unsigned N>
__global__ void BLOCK_REDUCE_KEPLER_TENSOR2(Tensor2CUDA<T,M,N> *in, Tensor2CUDA<T,M,N>* out, int nTerms) {
    Tensor2CUDA<T,M,N> sum;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<nTerms; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSumTensor2(sum);
    if (threadIdx.x==0) {
        out[blockIdx.x]=sum;
    }
}

template <typename T, unsigned int M, unsigned N>
void reduceKeplertTensor2(Tensor2CUDA<T,M,N> *inArray, Tensor2CUDA<T,M,N>* out, int nTerms, const cudaDeviceProp& prop) {
    // Get ideal configuration
    CudaKernelConfig reduceCfg = getKernelConfigMaxOccupancy(prop, BLOCK_REDUCE_KEPLER_TENSOR2<T,M,N>, nTerms);

    // Number of blocks
    unsigned int nBlocks = reduceCfg.dG.x;
    
    // Allocate memory for per-block sums
    std::shared_ptr<Tensor2CUDA<T,M,N>> perBlockSums = allocDeviceMemorySharedPtr<Tensor2CUDA<T,M,N>>(nBlocks);
    
    // Reduce in every block
    BLOCK_REDUCE_KEPLER<<<reduceCfg.dG, reduceCfg.dB>>>(inArray, perBlockSums.get(), N);

    // Reduce the resulting single block down
    if (nBlocks > prop.maxThreadsPerBlock) {
        ///@todo: Implement second-level reduction for sizes requiring more than maxThreadsPerBlock blocks
        std::cerr << "Warning: Reduction not yet implemented for that many blocks." << std::endl;
        std::cerr << "Warning: Defaulting to a single block reduction." << std::endl;
    }
    BLOCK_REDUCE_KEPLER<<<1, prop.maxThreadsPerBlock>>>(perBlockSums.get(), out, nBlocks);
}

/**
 * @brief Coresponds to the ZXZ Proper Euler Angles
 * @param R the rotation matrix
 * @return the angles
 */
template <typename T, unsigned int M, unsigned int N>
__device__ EulerAngles<T> getEulerZXZAngles(const Tensor2CUDA<T, M, N>& R)
{      
    EulerAngles<T> angle;
    
    // Angle beta
    angle.beta = acos(R.getVal(2,2));    
    
    // Nonsingular case
    float floatEpsilon = 1.19209e-07;
    if (angle.beta > 1e3*floatEpsilon) {
        // The other 2 angles
        angle.alpha = atan2(R.getVal(0,2),-R.getVal(1,2));
        angle.gamma = atan2(R.getVal(2,0),R.getVal(2,1));
    }
    // Singular case
    else {        
        angle.beta = 0.0;
        T alphaPlusGamma = atan2(-R.getVal(0,1), R.getVal(0,0));
        
        // Not uniquely determined, so just pick a combination
        angle.alpha = alphaPlusGamma/2.0;
        angle.gamma =  alphaPlusGamma/2.0;
    }
    
    // Correct the angle ranges if necessary
    if (angle.alpha < 0) angle.alpha += 2*(T)M_PI;
    if (angle.gamma < 0) angle.gamma += 2*(T)M_PI;
    
    // Return
    return angle;
}

///////////////////////
// TENSOR 2 SYMMETRIC//
///////////////////////

/**
 * @class Tensor2SymmCUDA
 * @author Michael Malahe
 * @date 04/05/17
 * @file tensorCUDA.h
 * @brief An anti-symmetric second order tensor
 * @details The underlying storage is row major. For example, for
 * a 4x4 symmetric tensor there will be a total of (4*(4+1))/2 = 10 elements
 * stored. Those elements are ordered in memory as:
 * [T(0,0),T(0,1),T(0,2),T(0,3),T(1,1),T(1,2),T(1,3),T(2,2),T(2,3),T(3,3)].
 */
template <typename T, unsigned int N>
class Tensor2SymmCUDA {
public:
    // Default constructor
    __host__ __device__ Tensor2SymmCUDA(){
        for (unsigned int i=0; i<this->getNelements(); i++) {
            vals[i] = (T)0.0;
        }
    }
    
    // Assignment from symmetric type
    __host__ __device__ Tensor2SymmCUDA<T,N>& operator=(const Tensor2SymmCUDA<T,N>& input) {
        memcpy(vals, input.vals, this->getNelements()*sizeof(T));
        return *this;
    }
    
    // Assignment from symmetric instance of arbitrary type    
    #ifdef __CUDA_ARCH__
        // Device version
        __device__ Tensor2SymmCUDA(const Tensor2CUDA<T,N,N>& input) {        
            // Assign values
            for (unsigned int i=0; i<N; i++) {
                for (unsigned int j=i; j<N; j++) {
                    this->setVal(i,j,input.getVal(i,j));
                }
            }
        }
    #else
        // Host version
        __host__ Tensor2SymmCUDA(const Tensor2CUDA<T,N,N>& input) {
            // Check that input is indeed anti-symmetric
            T closeEnough = 100*std::numeric_limits<T>::epsilon();
            for (unsigned int i=0; i<N; i++) {
                for (unsigned int j=i+1; j<N; j++) {
                    T val = input.getVal(i,j);
                    T symmVal = input.getVal(j,i);
                    if (std::abs(val-symmVal) > closeEnough) {
                        std::cerr << "(" << i << "," << j << ")=" << val << std::endl;
                        std::cerr << "(" << j << "," << i << ")=" << symmVal << std::endl;
                        throw std::runtime_error("Input tensor is not symmetric.");
                    }
                }
            }
            
            // Assign values
            for (unsigned int i=0; i<N; i++) {
                for (unsigned int j=i; j<N; j++) {
                    this->setVal(i,j,input.getVal(i,j));
                }
            }
        }        
    #endif
    
    // Assignment from anti-symmetric instance of arbitrary type
    __host__ Tensor2SymmCUDA(const Tensor2<T>& input) {
        // Use the Tensor2CUDA copy constructor
        Tensor2CUDA<T,N,N> inputCUDA = input;
        
        // Copy construct self from Tensor2CUDA
        *this = inputCUDA;
    }
    
    /**
     * @brief Get flat index for upper triangular portion.
     * @details Only valid inputs are j>=i
     * @param i
     * @param j
     * @return 
     */
    __host__ __device__ unsigned int getUpperFlatIdx(const unsigned int i, const unsigned int j) const {
        // Start with the index of the final element
        unsigned int idx = getNelements()-1;
        
        // Subtract the triangular numbers below and including our row
        idx -= ((N-i)*(N-i+1))/2;
        
        // Add our column offset
        idx += (j-i+1);
        
        ///@todo Is there a good way to flag an error here for invalid input?
        
        // Return
        return idx;
    }
    
    // Set values
    __host__ __device__ void setVal(const unsigned int i, const unsigned int j, const T val) {
        if (j>=i) {
            vals[getUpperFlatIdx(i,j)] = val;
        }
        else {
            vals[getUpperFlatIdx(j,i)] = val;
        }
    }
    
    // Get value
    __host__ __device__ T getVal(const unsigned int i, const unsigned int j) const {
        if (j>=i) {
            return vals[getUpperFlatIdx(i,j)];
        }
        else {
            return vals[getUpperFlatIdx(j,i)];
        }      
    }

    // Friends
    template <typename U, unsigned int M>
    friend __host__ __device__ Tensor2SymmCUDA<U,M> operator-(const Tensor2SymmCUDA<U,M>& A, const Tensor2SymmCUDA<U,M>& B);
    template <typename U, unsigned int M>
    friend __host__ __device__ Tensor2SymmCUDA<U,M> operator+(const Tensor2SymmCUDA<U,M>& A, const Tensor2SymmCUDA<U,M>& B);

protected:
    // Total number of elements in underlying storage
    __host__ __device__ unsigned int getNelements() const {
        return (N*(N+1))/2;
    }

    // Tensor values
    T vals[(N*(N+1))/2];
};

// Subtraction
template <typename T, unsigned int N>
__host__ __device__ Tensor2SymmCUDA<T,N> operator-(const Tensor2SymmCUDA<T,N>& A, const Tensor2SymmCUDA<T,N>& B) {
    Tensor2SymmCUDA<T,N> C;
    for (unsigned int idx=0; idx<A.getNelements(); idx++) {
        C.vals[idx] = A.vals[idx]-B.vals[idx];
    }
    return C;
}

// Addition
template <typename T, unsigned int N>
__host__ __device__ Tensor2SymmCUDA<T,N> operator+(const Tensor2SymmCUDA<T,N>& A, const Tensor2SymmCUDA<T,N>& B) {
    Tensor2SymmCUDA<T,N> C;
    for (unsigned int idx=0; idx<A.getNelements(); idx++) {
        C.vals[idx] = A.vals[idx]+B.vals[idx];
    }
    return C;
}

// Matrix multiplication
// Symmetric NxN times arbitrary NxP
template <typename T, unsigned int N, unsigned int P>
__host__ __device__ Tensor2CUDA<T,N,P> operator*(const Tensor2SymmCUDA<T,N>& A, const Tensor2CUDA<T,N,P>& B) {
    Tensor2CUDA<T,N,P> C;
    for (unsigned int i=0; i<N; i++) {
        for (unsigned int j=0; j<P; j++) {
            for (unsigned int k=0; k<N; k++) {
                C(i,j) += A.getVal(i,k)*B.getVal(k,j);
            }
        }
    }
    return C;
}

////////////////////////////
// TENSOR 2 ANTI-SYMMETRIC//
////////////////////////////

/**
 * @class Tensor2AsymmCUDA
 * @author Michael Malahe
 * @date 28/04/17
 * @file tensorCUDA.h
 * @brief An anti-symmetric second order tensor
 * @details The underlying storage is row major. For example, for
 * a 4x4 anti-symmetric tensor there will be a total of (4*(4-1))/2 = 6 elements
 * stored. Those elements are ordered in memory as:
 * [T(0,1),T(0,2),T(0,3),T(1,2),T(1,3),T(2,3)].
 */
template <typename T, unsigned int N>
class Tensor2AsymmCUDA {
public:
    // Default constructor
    __host__ __device__ Tensor2AsymmCUDA(){
        for (unsigned int i=0; i<this->getNelements(); i++) {
            vals[i] = (T)0.0;
        }
    }
    
    // Assignment from anti-symmetric type
    __host__ __device__ Tensor2AsymmCUDA<T,N>& operator=(const Tensor2AsymmCUDA<T,N>& input) {
        memcpy(vals, input.vals, this->getNelements()*sizeof(T));
        return *this;
    }
    
    // Assignment from anti-symmetric instance of arbitrary type
    __host__ Tensor2AsymmCUDA(const Tensor2CUDA<T,N,N>& input) {
        // Check that input is indeed anti-symmetric
        T closeEnough = 100*std::numeric_limits<T>::epsilon();
        for (unsigned int i=0; i<N; i++) {
            for (unsigned int j=i; j<N; j++) {
                T val = input.getVal(i,j);
                T asymmVal = input.getVal(j,i);
                if (std::abs(val+asymmVal) > closeEnough) {
                    std::cerr << "(" << i << "," << j << ")=" << val << std::endl;
                    std::cerr << "(" << j << "," << i << ")=" << asymmVal << std::endl;
                    throw std::runtime_error("Input tensor is not anti-symmetric.");
                }
            }
        }
        
        // Assign values
        for (unsigned int i=0; i<N-1; i++) {
            for (unsigned int j=i+1; j<N; j++) {
                this->setVal(i,j,input.getVal(i,j));
            }
        }
    }
    
    // Assignment from anti-symmetric instance of arbitrary type
    __host__ Tensor2AsymmCUDA(const Tensor2<T>& input) {
        // Use the Tensor2CUDA copy constructor
        Tensor2CUDA<T,N,N> inputCUDA = input;
        
        // Copy construct self from Tensor2CUDA
        *this = inputCUDA;
    }
    
    /**
     * @brief Get flat index for upper triangular portion.
     * @details Only valid inputs are j>i
     * @param i
     * @param j
     * @return 
     */
    __host__ __device__ unsigned int getUpperFlatIdx(const unsigned int i, const unsigned int j) const {
        // Start with the index of the final element
        unsigned int idx = getNelements()-1;
        
        // Subtract the triangular numbers below and including our row
        idx -= ((N-i-1)*(N-i))/2;
        
        // Add our column offset from the diagonal
        idx += (j-i);
        
        ///@todo Is there a good way to flag an error here for invalid input?
        
        // Return
        return idx;
    }
    
    // Get read/write reference to value
    __host__ __device__ void setVal(const unsigned int i, const unsigned int j, const T val) {
        if (j>i) {
            vals[getUpperFlatIdx(i,j)] = val;
        }
        else if (i>j) {
            vals[getUpperFlatIdx(j,i)] = -val;
        }
    }
    
    // Get read-only value
    __host__ __device__ T getVal(const unsigned int i, const unsigned int j) const {
        if (i==j) {
            return (T)0.0;
        }
        else {
            if (j>i) {
                return vals[getUpperFlatIdx(i,j)];
            }
            else {
                return -vals[getUpperFlatIdx(j,i)];
            }
        }        
    }

    // Friends
    template <typename U, unsigned int M>
    friend __host__ __device__ Tensor2AsymmCUDA<U,M> operator-(const Tensor2AsymmCUDA<U,M>& A, const Tensor2AsymmCUDA<U,M>& B);
    template <typename U, unsigned int M>
    friend __host__ __device__ Tensor2AsymmCUDA<U,M> operator+(const Tensor2AsymmCUDA<U,M>& A, const Tensor2AsymmCUDA<U,M>& B);

protected:
    // Total number of elements in underlying storage
    __host__ __device__ unsigned int getNelements() const {
        return (N*(N-1))/2;
    }

    // Tensor values
    T vals[(N*(N-1))/2];
};

// Subtraction
template <typename T, unsigned int N>
__host__ __device__ Tensor2AsymmCUDA<T,N> operator-(const Tensor2AsymmCUDA<T,N>& A, const Tensor2AsymmCUDA<T,N>& B) {
    Tensor2AsymmCUDA<T,N> C;
    for (unsigned int idx=0; idx<A.getNelements(); idx++) {
        C.vals[idx] = A.vals[idx]-B.vals[idx];
    }
    return C;
}

template <typename T, unsigned int N>
__host__ __device__ Tensor2CUDA<T,N,N> operator-(const Tensor2AsymmCUDA<T,N>& A, const Tensor2CUDA<T,N,N>& B) {
    Tensor2CUDA<T,N,N> C;
    for (unsigned int i=0; i<N; i++) {
        for (unsigned int j=0; j<N; j++) {
            C(i,j) = A.getVal(i,j) - B.getVal(i,j);
        }
    }
    return C;
}

// Addition
template <typename T, unsigned int N>
__host__ __device__ Tensor2AsymmCUDA<T,N> operator+(const Tensor2AsymmCUDA<T,N>& A, const Tensor2AsymmCUDA<T,N>& B) {
    Tensor2AsymmCUDA<T,N> C;
    for (unsigned int idx=0; idx<A.getNelements(); idx++) {
        C.vals[idx] = A.vals[idx]+B.vals[idx];
    }
    return C;
}

// Matrix multiplication
// Anti-symmetric NxN times arbitrary NxP
template <typename T, unsigned int N, unsigned int P>
__host__ __device__ Tensor2CUDA<T,N,P> operator*(const Tensor2AsymmCUDA<T,N>& A, const Tensor2CUDA<T,N,P>& B) {
    Tensor2CUDA<T,N,P> C;
    for (unsigned int i=0; i<N; i++) {
        for (unsigned int j=0; j<P; j++) {
            for (unsigned int k=0; k<N; k++) {
                C(i,j) += A.getVal(i,k)*B.getVal(k,j);
            }
        }
    }
    return C;
}

// Arbitrary NXP times anti-symmetric PxP
template <typename T, unsigned int N, unsigned int P>
__host__ __device__ Tensor2CUDA<T,N,P> operator*(const Tensor2CUDA<T,N,P>& A, const Tensor2AsymmCUDA<T,P>& B) {
    Tensor2CUDA<T,N,P> C;
    for (unsigned int i=0; i<N; i++) {
        for (unsigned int j=0; j<P; j++) {
            for (unsigned int k=0; k<P; k++) {
                C(i,j) += A.getVal(i,k)*B.getVal(k,j);
            }
        }
    }
    return C;
}

//////////////
// TENSOR 4 //
//////////////

template <typename U, unsigned int M, unsigned int N, unsigned int P, unsigned int Q>
class Tensor4CUDA {
public:
    Tensor4CUDA(){;}
    Tensor4CUDA(const Tensor4<U>& in);
private:
    U vals[M][N][P][Q];
};

// Constructor from a standard Tensor4
template <typename U, unsigned int M, unsigned int N, unsigned int P, unsigned int Q>
Tensor4CUDA<U,M,N,P,Q>::Tensor4CUDA(const Tensor4<U>& in) {
    // Check size
    if (M != in.n1 || N != in.n2 || P != in.n3 || Q != in.n4) {
        throw TensorError("Size mismatch.");
    }
    
    // Copy
    std::copy(&(in.vals[0]), &(in.vals[0])+M*N*P*Q, &(vals[0][0][0][0]));    
}

#endif /* HPP_USE_CUDA */
}//END NAMESPACE HPP

#endif /* TENSOR_CUDA_H */
