/// @file testTensor2.cpp
/// @author Michael Malahe
/// @brief Code for testing members of hpp::Tensor2
#include <hpp/tensor.h>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <limits>

namespace hpp {

/**
 * @brief Test the basic functions of Tensor2
 * @tparam the scalar type
 */
template <typename T>
void testTensor2Basics() {
    // Close enough to be considered equal for a floating point number
    T closeEnough = 10*std::numeric_limits<T>::epsilon();
    
    // Base tensors
    Tensor2<T> A(2,2);
    A(0,0) = 1.0;
    A(0,1) = -2.0;
    A(1,0) = -3.0;
    A(1,1) = 8.0;
    Tensor2<T> A_almost(2,2);
    T bump = 100.0*std::numeric_limits<T>::epsilon();
    A_almost = A + bump;
    
    // Equality
    if (!(A == A)) throw std::runtime_error("equality evaluation failed");
    if (!(A != A_almost)) throw std::runtime_error("inequality evaluation failed"); 
    
    // Inverse
    Tensor2<T> AInvAnalytic(2,2);
    AInvAnalytic(0,0) = 4.0;
    AInvAnalytic(0,1) = 1.0;
    AInvAnalytic(1,0) = 1.5;
    AInvAnalytic(1,1) = 0.5;
    if (!(A.inv() == AInvAnalytic)) throw std::runtime_error("inverse failed");
    
    // Inverse in place
    Tensor2<T> AInv = A;
    AInv.invInPlace();
    if (!(AInv == AInvAnalytic)) throw std::runtime_error("in-place inverse failed");

    // Trace
    T trAnalytic = 9.0;
    if (!(A.tr() == trAnalytic)) throw std::runtime_error("trace evaluation failed");
    
    // Determinant
    T detAnalytic = 2.0;
    if (!(std::abs(A.det()-detAnalytic) < closeEnough)) throw std::runtime_error("determinant evaluation failed");
    
    // Scaling to unit determinant
    if (!(std::abs(A.scaledToUnitDeterminant().det()-1.0) < closeEnough)) throw std::runtime_error("scaling to unit determinant failed");
    
    // Check error for determinant scaling of negative determinant tensor
    Tensor2<T> negativeDeterminantTensor(2,2);
    negativeDeterminantTensor(0,0) = 1.0;
    negativeDeterminantTensor(0,1) = 0.0;
    negativeDeterminantTensor(1,0) = 0.0;
    negativeDeterminantTensor(1,1) = -1.0;
    bool caughtNegativeDeterminantTensor = false;
    try {        
        negativeDeterminantTensor.scaledToUnitDeterminant();
    } catch (TensorError& e) {
        caughtNegativeDeterminantTensor = true;
    }
    if (!(caughtNegativeDeterminantTensor)) throw std::runtime_error("negative determinant tensor error was not raised correctly");
    
    // Transpose
    Tensor2<T> ATAnalytic(2,2);
    ATAnalytic(0,0) = 1.0;
    ATAnalytic(0,1) = -3.0;
    ATAnalytic(1,0) = -2.0;
    ATAnalytic(1,1) = 8.0;
    if (!(A.trans() == ATAnalytic)) throw std::runtime_error("transpose failed");
    
    // Min and max
    T minAnalytic = -3.0;
    T maxAnalytic = 8.0;
    if (!(A.min() == minAnalytic)) throw std::runtime_error("min failed");
    if (!(A.max() == maxAnalytic)) throw std::runtime_error("max failed");
    
    // Abs
    Tensor2<T> AAbsAnalytic(2,2);
    AAbsAnalytic(0,0) = 1.0;
    AAbsAnalytic(0,1) = 2.0;
    AAbsAnalytic(1,0) = 3.0;
    AAbsAnalytic(1,1) = 8.0;
    if (!(A.abs() == AAbsAnalytic)) throw std::runtime_error("abs failed");
    
    // Constraining
    T constrainMin = -2.5;
    T constrainMax = 7.5;
    Tensor2<T> AConstrainedAnalytic(2,2);
    AConstrainedAnalytic(0,0) = 1.0;
    AConstrainedAnalytic(0,1) = -2.0;
    AConstrainedAnalytic(1,0) = -2.5;
    AConstrainedAnalytic(1,1) = 7.5;
    if (!(A.constrainedTo(constrainMin, constrainMax) == AConstrainedAnalytic)) throw std::runtime_error("constraining failed");
    
    // Poorly-formed constraint
    bool caughtBadConstraint = false;
    try {        
        A.constrainedTo(1.0,-1.0);
    } catch (TensorError& e) {
        caughtBadConstraint = true;
    }
    if (!(caughtBadConstraint)) throw std::runtime_error("bad constraint error was not raised correctly");
    
    // Check errors for non-square tensors
    Tensor2<T> notSquare(1,2);
    
    // Check square assertion
    bool caughtNotSquare = false;
    try {        
        notSquare.assertSquare();
    } catch (TensorError& e) {
        caughtNotSquare = true;
    }
    if (!(caughtNotSquare)) throw std::runtime_error("non-square tensor error was not raised correctly");
    
    // Check error thrown for non-square inverse
    bool caughtNonSquareInverse = false;
    try {        
        notSquare.inv();
    } catch (TensorError& e) {
        caughtNonSquareInverse = true;
    }
    if (!(caughtNonSquareInverse)) throw std::runtime_error("Non-square tensor error was not raised correctly.");
    
    // Check error thrown for singular matrix
    Tensor2<T> singularTensor(2,2);
    singularTensor(0,0) = 1.0;
    singularTensor(0,1) = 2.0;
    singularTensor(1,0) = 0.5;
    singularTensor(1,1) = 1.0;
    bool caughtSingularTensor = false;
    try {        
        singularTensor.inv();
    } catch (TensorError& e) {
        caughtSingularTensor = true;
    }
    if (!(caughtSingularTensor)) throw std::runtime_error("Singular tensor error was not raised correctly.");
}

template<typename T>
void testTensor2BinaryOperations() {
    // Re-used variables
    T one = 1.0;
    T two = 2.0;
    
    // Base tensors
    Tensor2<T> A(2,2);
    A(0,0) = 1.0;
    A(0,1) = 2.0;
    A(1,0) = 3.0;
    A(1,1) = 4.0;
    Tensor2<T> B(2,2);
    B(0,0) = 2.0;
    B(0,1) = 0.0;
    B(1,0) = 0.0;
    B(1,1) = 2.0;
    Tensor2<T> ADifferentShape(1,2);
    
    // Addition
    Tensor2<T> APlusOne(2,2);
    APlusOne(0,0) = 2.0;
    APlusOne(0,1) = 3.0;
    APlusOne(1,0) = 4.0;
    APlusOne(1,1) = 5.0;
    if (!(A+one == APlusOne)) throw std::runtime_error("Addition failed.");
    if (!(one+A == APlusOne)) throw std::runtime_error("Addition failed.");
    Tensor2<T> APlusB(2,2);
    APlusB(0,0) = 3.0;
    APlusB(0,1) = 2.0;
    APlusB(1,0) = 3.0;
    APlusB(1,1) = 6.0;
    if (!(A+B == APlusB)) throw std::runtime_error("Addition failed.");
    Tensor2<T> AIncrementB(A);
    AIncrementB += B;
    if (!(AIncrementB == APlusB)) throw std::runtime_error("Addition failed.");
    
    // Addition size mismatch
    bool caughtIncompatibleTensor = false;
    try {        
        A + ADifferentShape;
    } catch (TensorError& e) {
        caughtIncompatibleTensor = true;
    }
    if (!(caughtIncompatibleTensor)) throw std::runtime_error("Incompatible tensor addition error was not raised correctly.");
    
    // Subtraction
    Tensor2<T> AMinusOne(2,2);
    AMinusOne(0,0) = 0.0;
    AMinusOne(0,1) = 1.0;
    AMinusOne(1,0) = 2.0;
    AMinusOne(1,1) = 3.0;
    if (!(A-one == AMinusOne)) throw std::runtime_error("Subtraction failed.");
    if (!(one-A == AMinusOne)) throw std::runtime_error("Subtraction failed.");
    Tensor2<T> AMinusB(2,2);
    AMinusB(0,0) = -1.0;
    AMinusB(0,1) = 2.0;
    AMinusB(1,0) = 3.0;
    AMinusB(1,1) = 2.0;
    if (!(A-B == AMinusB)) throw std::runtime_error("Subtraction failed.");
    Tensor2<T> ADecrementB(A);
    ADecrementB -= B;
    if (!(ADecrementB == AMinusB)) throw std::runtime_error("Subtraction failed.");
    
    // Addition size mismatch
    caughtIncompatibleTensor = false;
    try {        
        A - ADifferentShape;
    } catch (TensorError& e) {
        caughtIncompatibleTensor = true;
    }
    if (!(caughtIncompatibleTensor)) throw std::runtime_error("Incompatible tensor subtraction error was not raised correctly.");
    
    // Scalar multiplication
    Tensor2<T> ATimesTwo(2,2);
    ATimesTwo(0,0) = 2.0;
    ATimesTwo(0,1) = 4.0;
    ATimesTwo(1,0) = 6.0;
    ATimesTwo(1,1) = 8.0;
    if (!(A*two == ATimesTwo)) throw std::runtime_error("Scalar multiplication failed.");
    if (!(two*A == ATimesTwo)) throw std::runtime_error("Scalar multiplication failed.");
    Tensor2<T> ATimesEqualsTwo(A);
    ATimesEqualsTwo *= two;
    if (!(A*two == ATimesEqualsTwo)) throw std::runtime_error("Scalar multiplication failed.");
    
    // Scalar Division
    Tensor2<T> ADividedByTwo(2,2);
    ADividedByTwo(0,0) = 0.5;
    ADividedByTwo(0,1) = 1.0;
    ADividedByTwo(1,0) = 1.5;
    ADividedByTwo(1,1) = 2.0;
    if (!(A/two == ADividedByTwo)) throw std::runtime_error("Scalar division failed.");
    Tensor2<T> ADividedByEqualsTwo(A);
    ADividedByEqualsTwo /= two;
    if (!(A/two == ADividedByEqualsTwo)) throw std::runtime_error("Scalar division failed.");
    
    // Tensor multiplication
    Tensor2<T> ATimesB(2,2);
    ATimesB(0,0) = 2.0;
    ATimesB(0,1) = 4.0;
    ATimesB(1,0) = 6.0;
    ATimesB(1,1) = 8.0;
    if (!(A*B == ATimesB)) throw std::runtime_error("Tensor multiplication failed.");
    
    // Identity tensor
    Tensor2<T> I2 = identityTensor2<T>(2);
    if (!(I2*A == A)) throw std::runtime_error("identity tensor failed");
    
    // Tensor multiplication size mismatch
    Tensor2<T> incompatibleTensor(2,3);
    caughtIncompatibleTensor = false;
    try {        
        incompatibleTensor*A;
    } catch (TensorError& e) {
        caughtIncompatibleTensor = true;
    }
    if (!(caughtIncompatibleTensor)) throw std::runtime_error("Incompatible tensor multiplication error was not raised correctly.");
    
    // Tensor contraction
    T AContractB = 10.0;
    if (!(contract(A,B) == AContractB)) throw std::runtime_error("Tensor contraction failed.");
    
    // Tensor contraction size mismatch
    caughtIncompatibleTensor = false;
    try {        
        contract(incompatibleTensor,A);
    } catch (TensorError& e) {
        caughtIncompatibleTensor = true;
    }
    if (!(caughtIncompatibleTensor)) throw std::runtime_error("Incompatible tensor contraction error was not raised correctly.");
}

template<typename T>
void testTensor2Decompositions() {
    // TESTS WITH SYMMETRIC MATRIX
    
    // Base tensor
    Tensor2<T> A(2,2);
    A(0,0) = 1.0;
    A(0,1) = -2.0;
    A(1,0) = -2.0;
    A(1,1) = 8.0;
    
    // Square root
    Tensor2<T> sqrtA = A.sqrtMatrix();
    Tensor2<T> sqrtASquared = sqrtA*sqrtA;
    if (sqrtASquared != A) {
        std::cout << "A = " << A << std::endl;
        std::cout << "sqrtASquared = " << sqrtASquared << std::endl;
        throw TensorError("Matrix square root failed.");
    }
    
    // TESTS FOR NOT NONSYMMETRIC MATRIX
    
    // Base tensor
    Tensor2<T> B(2,2);
    B(0,0) = 1.0;
    B(0,1) = -2.0;
    B(1,0) = -3.0;
    B(1,1) = 8.0;
    
    // Polar decomposition
    T polarCloseEnough = 100*std::numeric_limits<T>::epsilon();
    PolarDecomposition<T> decomp = B.polarDecomposition();
    Tensor2<T> B_recomposed = decomp.R*decomp.U;
    if (!areEqual(B, B_recomposed, polarCloseEnough)) {
        std::cout << "B = " << B << std::endl;
        std::cout << "B_recomposed = " << B_recomposed << std::endl;
        throw TensorError("Polar decomposition failed.");
    }
    
    // Rotations
    EulerAngles<T> angle;
    T angleEquiv = 1e6*std::numeric_limits<T>::epsilon();
    int NPointsPerAngle = 16;
    for (int i=0; i<NPointsPerAngle; i++) {
        angle.alpha = i*(2*M_PI/NPointsPerAngle);
        for (int j=0; j<NPointsPerAngle; j++) {
            angle.beta = j*(M_PI/NPointsPerAngle)+0.001;
            for (int k=0; k<NPointsPerAngle; k++) {
                angle.gamma = k*(2*M_PI/NPointsPerAngle);
                Tensor2<T> R = EulerZXZRotationMatrix(angle);
                EulerAngles<T> angleRestored = getEulerZXZAngles(R);
                bool angleGood = true;
                if (std::abs(angle.alpha-angleRestored.alpha) > angleEquiv) angleGood = false;
                if (std::abs(angle.beta-angleRestored.beta) > angleEquiv) angleGood = false;
                if (std::abs(angle.gamma-angleRestored.gamma) > angleEquiv) angleGood = false;
                if (!angleGood) {
                    std::cerr << "Input: " << angle.alpha << " " << angle.beta << " " << angle.gamma << std::endl;
                    std::cerr << "Output: " << angleRestored.alpha << " " << angleRestored.beta << " " << angleRestored.gamma << std::endl;
                    throw TensorError("Angles don't match.");
                }
            }
        }
    }  
}

template<typename T> void getAxisAngle(Tensor2<T> R, std::vector<T>& u, T& theta){
    u[0] = R(2,1)-R(1,2);
    u[1] = R(0,2)-R(2,0);
    u[2] = R(1,0)-R(0,1);
    T r = std::sqrt(std::pow(u[0],(T)2.0)+std::pow(u[1],(T)2.0)+std::pow(u[2],(T)2.0));
    T t = R.tr();
    theta = std::atan2(r, t-1);
}

template<typename T>
void testTensor2Random() {
    // Do an axis-angle decomposition, and ensure that the axis is uniformly
    // distributed, and the angle is distributed with CDF (1/pi)(theta-sin(theta))
    // i.e. PDF (1/pi)(1-cos(theta))
    Tensor2<T> R(3,3);
    std::vector<T> u(3);
    T theta;
    
    // Set up histogram
    int nBins = 100;
    int nSamples = 1000000;
    T binWidthTheta = M_PI/(nBins);
    std::valarray<T> histTheta(nBins);
    histTheta = 0.0;
    
    // Samples
    for (unsigned int iS=0; iS<nSamples; iS++) {
        randomRotationTensorInPlace<T>(3, R, true);
        getAxisAngle(R, u, theta);
        int iTheta = (int)(theta/binWidthTheta);
        histTheta[iTheta] += 1.0;
    }
    histTheta /= (T)nSamples;
    
    // Analytic density
    std::valarray<T> histAnalyticTheta(nBins);
    for (unsigned int iBin=0; iBin<nBins; iBin++) {
        theta = (iBin+0.5)*binWidthTheta;
        histAnalyticTheta[iBin] = binWidthTheta*(1.0/M_PI)*(1.0-std::cos(theta));
    }
    
    // Theta error
    std::valarray<T> histErrorTheta = std::abs(histTheta-histAnalyticTheta)/histAnalyticTheta;
    T meanErrorTheta = 0.0;
    for (auto err : histErrorTheta) {
        meanErrorTheta += err;
    }
    meanErrorTheta /= histErrorTheta.size();
    if (meanErrorTheta > 0.03) {
        std::cerr << "Mean error: " << meanErrorTheta << std::endl;
        throw TensorError("Mean error in random rotation is too large.");
    }
}

template<typename T>
void testTensor2HDF5() {
    // SINGLE TENSOR HDF5 TEST
    
    // Create tensor and names
    std::vector<hsize_t> tensorDims = {2,2};
    Tensor2<T> tensorOut(tensorDims[0], tensorDims[1]);
    tensorOut(0,0) = 1.0;
    tensorOut(0,1) = 2.0;
    tensorOut(1,0) = 3.0;
    tensorOut(1,1) = 4.0;
    H5std_string filename = "serialOutput.hdf5";
    H5std_string datasetName = "tensor";
    
    // Write tensor out
    tensorOut.writeToNewHDF5(filename, datasetName);
    
    // Open file back up and read
    H5::H5File file(filename.c_str(), H5F_ACC_RDWR);
    Tensor2<T> tensorIn(file, datasetName, tensorDims);
    file.close();
    
    // Check for equality
    if (tensorIn != tensorOut) {
        throw TensorError("Tensor written isn't the same as tensor read.");
    }
    
    // MULTIPLE TENSOR HDF5 TEST
    file = H5::H5File(filename.c_str(), H5F_ACC_TRUNC);
    
    // Create an array of 2x2 tensors
    std::vector<hsize_t> tensorArraySize = {4,3};
    std::vector<std::vector<Tensor2<T>>> tensorArray2D(tensorArraySize[0]);
    for (auto&& tensorArray1D : tensorArray2D) {
        tensorArray1D = std::vector<Tensor2<T>>(tensorArraySize[1]);
        for (auto&& eachTensor : tensorArray1D) {
            eachTensor = Tensor2<T>(2,2);
        }
    }
    
    // Populate it arbitrarily
    for (unsigned int i=0; i<tensorArraySize[0]; i++) {
        for (unsigned int j=0; j<tensorArraySize[1]; j++) {
            tensorArray2D[i][j](0,0) = 1.0*i;
            tensorArray2D[i][j](0,1) = 2.0*i;
            tensorArray2D[i][j](1,0) = 3.0*j;
            tensorArray2D[i][j](1,1) = 4.0*j;
        }
    }
    
    // Create HDF5 dataset
    H5::DataSet dataset = createHDF5GridOfArrays<T>(file, datasetName, tensorArraySize, tensorDims);
    
    // Write to HDF5 file
    for (unsigned int i=0; i<tensorArraySize[0]; i++) {
        for (unsigned int j=0; j<tensorArraySize[1]; j++) {
            std::vector<hsize_t> arrayOffset = {i,j};
            tensorArray2D[i][j].writeToExistingHDF5Dataset(dataset, arrayOffset);
        }
    }
    
    // Close file
    file.close();
    
    // Open for reading
    file = H5::H5File(filename.c_str(), H5F_ACC_RDWR);
    dataset = file.openDataSet(datasetName.c_str());
    
    // Read back in and compare
    for (unsigned int i=0; i<tensorArraySize[0]; i++) {
        for (unsigned int j=0; j<tensorArraySize[1]; j++) {
            std::vector<hsize_t> arrayOffset = {i,j};
            Tensor2<T> tensorIn(dataset, arrayOffset, tensorDims);
            if (tensorIn != tensorArray2D[i][j]) {
                throw TensorError("Tensor written isn't the same as tensor read.");
            }
        }
    }

    // Close file
    file.close();    
}

} //END NAMESPACE TENSOR

int main(int argc, char *argv[]) {
    hpp::testTensor2Basics<float>();
    hpp::testTensor2Basics<double>();
    hpp::testTensor2BinaryOperations<float>();
    hpp::testTensor2BinaryOperations<double>();
    hpp::testTensor2Decompositions<float>();
    hpp::testTensor2Decompositions<double>();
    hpp::testTensor2Random<float>();
    hpp::testTensor2Random<double>();
    hpp::testTensor2HDF5<float>();  
    hpp::testTensor2HDF5<double>();  
    return 0;
}

