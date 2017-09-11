/** @file testTensor.cpp
* @author Michael Malahe
* @brief Tests for functions in tensor.h.
* @details Does not include behaviour limited to individual tensor orders. Those
* can be found in testTensor2.cpp and testTensor4.cpp;
*/

#include <hpp/tensor.h>
#include <cassert>
#include <iostream>
#include <limits>

namespace hpp {

template<typename T>
void testTensor4Tensor2Interactions() {    
    // Base tensors
    Tensor4<T> A4(2,2,2,2);
    A4(0,0,0,0) = -1.0;
    A4(0,1,0,0) = 2.0;
    A4(1,0,0,0) = -3.0;
    A4(1,1,0,0) = 4.0;
    A4(0,0,0,1) = -5.0;
    A4(0,1,0,1) = 6.0;
    A4(1,0,0,1) = -7.0;
    A4(1,1,0,1) = 8.0;
    A4(0,0,1,0) = -9.0;
    A4(0,1,1,0) = 10.0;
    A4(1,0,1,0) = -11.0;
    A4(1,1,1,0) = 12.0;
    A4(0,0,1,1) = -13.0;
    A4(0,1,1,1) = 14.0;
    A4(1,0,1,1) = -15.0;
    A4(1,1,1,1) = 16.0;
    Tensor2<T> A2(2,2);
    A2(0,0) = 2.0;
    A2(0,1) = 0.0;
    A2(1,0) = 0.0;
    A2(1,1) = 2.0;
    
    // Order conversions
    assert(Tensor4<T>(Tensor2<T>(A4)) == A4 && "Order conversion failed.");
    
    // Contraction
    Tensor2<T> A4ContractA2(2,2);
    A4ContractA2(0,0) = -28.0;
    A4ContractA2(0,1) = 32.0;
    A4ContractA2(1,0) = -36.0;
    A4ContractA2(1,1) = 40.0;
    assert(contract(A4,A2) == A4ContractA2 && "Contraction evaluation failed.");
    
    // Contraction incompatability
    Tensor2<T> A2ContractionIncompatible(3,2);
    bool caughtContractionIncompatible = false;
    try {        
        contract(A4, A2ContractionIncompatible);
    } catch (TensorError e) {
        caughtContractionIncompatible = true;
    }
    assert(caughtContractionIncompatible && "Incomparible tensor contraction was not raised correctly.");
    
    // Fourth order identity
    Tensor4<T> I4 = identityTensor4<T>(2);
    assert(contract(I4, A2) == A2 && "Fourth order identity failed.");
    
    // Fourth order inverse
    // A4 was just constructed randomly
    //
    T closeEnoughConditionNumberFudge = 1000*std::numeric_limits<T>::epsilon(); ///@todo yeah...
    Tensor4<T> A4Nonsingular(2,2,2,2);
    A4Nonsingular(0,0,0,0) = 0.05;
    A4Nonsingular(0,1,0,0) = 0.95;
    A4Nonsingular(1,0,0,0) = 0.36;
    A4Nonsingular(1,1,0,0) = 0.77;
    A4Nonsingular(0,0,0,1) = 0.18;
    A4Nonsingular(0,1,0,1) = 0.02;
    A4Nonsingular(1,0,0,1) = 0.45;
    A4Nonsingular(1,1,0,1) = 0.12;
    A4Nonsingular(0,0,1,0) = 0.39;
    A4Nonsingular(0,1,1,0) = 0.64;
    A4Nonsingular(1,0,1,0) = 0.36;
    A4Nonsingular(1,1,1,0) = 0.22;
    A4Nonsingular(0,0,1,1) = 0.20;
    A4Nonsingular(0,1,1,1) = 0.21;
    A4Nonsingular(1,0,1,1) = 0.65;
    A4Nonsingular(1,1,1,1) = 0.74;
    Tensor4<T> A4NonsingularInv = A4Nonsingular.inv();
    assert(((contract(A4NonsingularInv,contract(A4Nonsingular, A2)) - A2).abs()).max() < closeEnoughConditionNumberFudge && "Fourth order inverse failed.");
    assert(((contract(A4Nonsingular,contract(A4NonsingularInv, A2)) - A2).abs()).max() < closeEnoughConditionNumberFudge && "Fourth order inverse failed.");
    
    // In-place inverse
    A4NonsingularInv = A4Nonsingular;
    A4NonsingularInv.invInPlace();
    assert(((contract(A4NonsingularInv,contract(A4Nonsingular, A2)) - A2).abs()).max() < closeEnoughConditionNumberFudge && "Fourth order inverse failed.");
    assert(((contract(A4Nonsingular,contract(A4NonsingularInv, A2)) - A2).abs()).max() < closeEnoughConditionNumberFudge && "Fourth order inverse failed.");    
    
    // Outer product
    Tensor2<T> C2(2,2);
    C2(0,0) = 1.0;
    C2(0,1) = 2.0;
    C2(1,0) = 3.0;
    C2(1,1) = 4.0;
    Tensor2<T> B2(2,2);
    B2(0,0) = 2.0;
    B2(0,1) = 3.0;
    B2(1,0) = 4.0;
    B2(1,1) = 5.0;
    Tensor4<T> C2OuterB2(2,2,2,2);
    C2OuterB2(0,0,0,0) = 2.0;
    C2OuterB2(0,1,0,0) = 4.0;
    C2OuterB2(1,0,0,0) = 6.0;
    C2OuterB2(1,1,0,0) = 8.0;
    C2OuterB2(0,0,0,1) = 3.0;
    C2OuterB2(0,1,0,1) = 6.0;
    C2OuterB2(1,0,0,1) = 9.0;
    C2OuterB2(1,1,0,1) = 12.0;
    C2OuterB2(0,0,1,0) = 4.0;
    C2OuterB2(0,1,1,0) = 8.0;
    C2OuterB2(1,0,1,0) = 12.0;
    C2OuterB2(1,1,1,0) = 16.0;
    C2OuterB2(0,0,1,1) = 5.0;
    C2OuterB2(0,1,1,1) = 10.0;
    C2OuterB2(1,0,1,1) = 15.0;
    C2OuterB2(1,1,1,1) = 20.0;
    assert(outer(C2, B2) ==  C2OuterB2 && "Outer product failed.");
}

} //end namespace hpp

int main(int argc, char *argv[]) {
    hpp::testTensor4Tensor2Interactions<float>();
    hpp::testTensor4Tensor2Interactions<double>();  
    return 0;
}

