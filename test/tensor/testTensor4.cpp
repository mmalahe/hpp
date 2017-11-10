/// @file testTensor4.cpp
/// @author Michael Malahe
/// @brief Code for testing members of hpp::Tensor4
#include <hpp/tensor.h>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <limits>

namespace hpp {

/**
 * @brief Test the basic functions of Tensor4
 * @tparam the scalar type
 */
template <typename T>
void testTensor4Basics() {
    // Base tensors
    Tensor4<T> A(2,2,2,2);
    A(0,0,0,0) = -1.0;
    A(0,1,0,0) = 2.0;
    A(1,0,0,0) = -3.0;
    A(1,1,0,0) = 4.0;
    A(0,0,0,1) = -5.0;
    A(0,1,0,1) = 6.0;
    A(1,0,0,1) = -7.0;
    A(1,1,0,1) = 8.0;
    A(0,0,1,0) = -9.0;
    A(0,1,1,0) = 10.0;
    A(1,0,1,0) = -11.0;
    A(1,1,1,0) = 12.0;
    A(0,0,1,1) = -13.0;
    A(0,1,1,1) = 14.0;
    A(1,0,1,1) = -15.0;
    A(1,1,1,1) = 16.0;
    
    // Check unary operators
    Tensor4<T> negA(2,2,2,2);
    negA(0,0,0,0) = 1.0;
    negA(0,1,0,0) = -2.0;
    negA(1,0,0,0) = 3.0;
    negA(1,1,0,0) = -4.0;
    negA(0,0,0,1) = 5.0;
    negA(0,1,0,1) = -6.0;
    negA(1,0,0,1) = 7.0;
    negA(1,1,0,1) = -8.0;
    negA(0,0,1,0) = 9.0;
    negA(0,1,1,0) = -10.0;
    negA(1,0,1,0) = 11.0;
    negA(1,1,1,0) = -12.0;
    negA(0,0,1,1) = 13.0;
    negA(0,1,1,1) = -14.0;
    negA(1,0,1,1) = 15.0;
    negA(1,1,1,1) = -16.0;
    if (!(-A == negA)) throw std::runtime_error("Negation failed.");
    
    // Check errors for non-square tensors
    Tensor4<T> notSquare(1,2,3,4);
    
    // Check error thrown for non-square inverse
    bool caughtNonSquareInverse = false;
    try {        
        notSquare.inv();
    } catch (TensorError& e) {
        caughtNonSquareInverse = true;
    }
    if (!(caughtNonSquareInverse)) throw std::runtime_error("Non-square tensor error was not raised correctly.");
    
    // Check error thrown for singular tensor
    // Make a tensor with all zeros
    Tensor4<T> singularTensor(2,2,2,2);
    bool caughtSingularTensor = false;
    try {        
        singularTensor.inv();
    } catch (TensorError& e) {
        caughtSingularTensor = true;
    }
    if (!(caughtSingularTensor)) throw std::runtime_error("Singular tensor error was not raised correctly.");
}

template<typename T>
void testTensor4BinaryOperations() {
    // Re-used variables
    T one = 1.0;
    T two = 2.0;
    
    // Base tensors
    Tensor4<T> A(2,2,2,2);
    A(0,0,0,0) = -1.0;
    A(0,1,0,0) = 2.0;
    A(1,0,0,0) = -3.0;
    A(1,1,0,0) = 4.0;
    A(0,0,0,1) = -5.0;
    A(0,1,0,1) = 6.0;
    A(1,0,0,1) = -7.0;
    A(1,1,0,1) = 8.0;
    A(0,0,1,0) = -9.0;
    A(0,1,1,0) = 10.0;
    A(1,0,1,0) = -11.0;
    A(1,1,1,0) = 12.0;
    A(0,0,1,1) = -13.0;
    A(0,1,1,1) = 14.0;
    A(1,0,1,1) = -15.0;
    A(1,1,1,1) = 16.0;
    Tensor4<T> B(2,2,2,2);
    B(0,0,0,0) = 2.0;
    B(1,1,0,0) = 2.0;
    B(0,0,1,1) = 2.0;
    B(1,1,1,1) = 2.0;
    Tensor4<T> ADifferentShape(1,2,3,4);
    
    // Addition
    Tensor4<T> APlusOne(2,2,2,2);
    APlusOne(0,0,0,0) = 0.0;
    APlusOne(0,1,0,0) = 3.0;
    APlusOne(1,0,0,0) = -2.0;
    APlusOne(1,1,0,0) = 5.0;
    APlusOne(0,0,0,1) = -4.0;
    APlusOne(0,1,0,1) = 7.0;
    APlusOne(1,0,0,1) = -6.0;
    APlusOne(1,1,0,1) = 9.0;
    APlusOne(0,0,1,0) = -8.0;
    APlusOne(0,1,1,0) = 11.0;
    APlusOne(1,0,1,0) = -10.0;
    APlusOne(1,1,1,0) = 13.0;
    APlusOne(0,0,1,1) = -12.0;
    APlusOne(0,1,1,1) = 15.0;
    APlusOne(1,0,1,1) = -14.0;
    APlusOne(1,1,1,1) = 17.0;
    if (!(A+one == APlusOne)) throw std::runtime_error("Addition failed.");
    if (!(one+A == APlusOne)) throw std::runtime_error("Addition failed.");
    Tensor4<T> APlusB = A;
    APlusB(0,0,0,0) = 1.0;
    APlusB(1,1,0,0) = 6.0;
    APlusB(0,0,1,1) = -11.0;
    APlusB(1,1,1,1) = 18.0;
    if (!(A+B == APlusB)) throw std::runtime_error("Addition failed.");
    Tensor4<T> AIncrementB(A);
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
    Tensor4<T> AMinusOne(2,2,2,2);
    AMinusOne(0,0,0,0) = -2.0;
    AMinusOne(0,1,0,0) = 1.0;
    AMinusOne(1,0,0,0) = -4.0;
    AMinusOne(1,1,0,0) = 3.0;
    AMinusOne(0,0,0,1) = -6.0;
    AMinusOne(0,1,0,1) = 5.0;
    AMinusOne(1,0,0,1) = -8.0;
    AMinusOne(1,1,0,1) = 7.0;
    AMinusOne(0,0,1,0) = -10.0;
    AMinusOne(0,1,1,0) = 9.0;
    AMinusOne(1,0,1,0) = -12.0;
    AMinusOne(1,1,1,0) = 11.0;
    AMinusOne(0,0,1,1) = -14.0;
    AMinusOne(0,1,1,1) = 13.0;
    AMinusOne(1,0,1,1) = -16.0;
    AMinusOne(1,1,1,1) = 15.0;
    if (!(A-one == AMinusOne)) throw std::runtime_error("Subtraction failed.");
    if (!(one-A == AMinusOne)) throw std::runtime_error("Subtraction failed.");
    Tensor4<T> AMinusB = A;
    AMinusB(0,0,0,0) = -3.0;
    AMinusB(1,1,0,0) = 2.0;
    AMinusB(0,0,1,1) = -15.0;
    AMinusB(1,1,1,1) = 14.0;
    if (!(A-B == AMinusB)) throw std::runtime_error("Subtraction failed.");
    Tensor4<T> ADecrementB = A;
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
    Tensor4<T> ATimesTwo(2,2,2,2);
    ATimesTwo(0,0,0,0) = -2.0;
    ATimesTwo(0,1,0,0) = 4.0;
    ATimesTwo(1,0,0,0) = -6.0;
    ATimesTwo(1,1,0,0) = 8.0;
    ATimesTwo(0,0,0,1) = -10.0;
    ATimesTwo(0,1,0,1) = 12.0;
    ATimesTwo(1,0,0,1) = -14.0;
    ATimesTwo(1,1,0,1) = 16.0;
    ATimesTwo(0,0,1,0) = -18.0;
    ATimesTwo(0,1,1,0) = 20.0;
    ATimesTwo(1,0,1,0) = -22.0;
    ATimesTwo(1,1,1,0) = 24.0;
    ATimesTwo(0,0,1,1) = -26.0;
    ATimesTwo(0,1,1,1) = 28.0;
    ATimesTwo(1,0,1,1) = -30.0;
    ATimesTwo(1,1,1,1) = 32.0;
    if (!(A*two == ATimesTwo)) throw std::runtime_error("Scalar multiplication failed.");
    if (!(two*A == ATimesTwo)) throw std::runtime_error("Scalar multiplication failed.");
    Tensor4<T> ATimesEqualsTwo(A);
    ATimesEqualsTwo *= two;
    if (!(A*two == ATimesEqualsTwo)) throw std::runtime_error("Scalar multiplication failed.");
    
    // Scalar Division
    Tensor4<T> ADividedByTwo(2,2,2,2);
    ADividedByTwo(0,0,0,0) = -0.5;
    ADividedByTwo(0,1,0,0) = 1.0;
    ADividedByTwo(1,0,0,0) = -1.5;
    ADividedByTwo(1,1,0,0) = 2.0;
    ADividedByTwo(0,0,0,1) = -2.5;
    ADividedByTwo(0,1,0,1) = 3.0;
    ADividedByTwo(1,0,0,1) = -3.5;
    ADividedByTwo(1,1,0,1) = 4.0;
    ADividedByTwo(0,0,1,0) = -4.5;
    ADividedByTwo(0,1,1,0) = 5.0;
    ADividedByTwo(1,0,1,0) = -5.5;
    ADividedByTwo(1,1,1,0) = 6.0;
    ADividedByTwo(0,0,1,1) = -6.5;
    ADividedByTwo(0,1,1,1) = 7.0;
    ADividedByTwo(1,0,1,1) = -7.5;
    ADividedByTwo(1,1,1,1) = 8.0;
    if (!(A/two == ADividedByTwo)) throw std::runtime_error("Scalar division failed.");
    Tensor4<T> ADividedByEqualsTwo(A);
    ADividedByEqualsTwo /= two;
    if (!(A/two == ADividedByEqualsTwo)) throw std::runtime_error("Scalar division failed.");
    
    // Contraction size mismatch
    
    // Contraction
    
}

} //end namespace hpp

int main(int argc, char *argv[]) {
    hpp::testTensor4Basics<float>();
    hpp::testTensor4Basics<double>();
    hpp::testTensor4BinaryOperations<float>();
    hpp::testTensor4BinaryOperations<double>();   
    return 0;
}

