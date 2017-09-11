""" Tests for the linearAlgebra library
"""

import unittest
import numpy
from numpy import tensordot, array
from numpy.random import rand
from numpy.linalg import cond, norm

import sys
sys.path.append("..")
from linearAlgebra import *

MACHINE_PRECISION = 1e-15

class TestArrayBasics(unittest.TestCase):
    
    def test_ConstrainComponents(self):
        """Test the array component constraint functions.
        
        The functions covered are:
        - linearAlgebra.constrainComponents3by3
        - linearAlgebra.constrainComponents3by3InPlace
        """
        A = array([[4.0,-4.0,4.0],
                   [-5.0,5.0,-5.0],
                   [-6.0,6.0,-6.0]])
        A_max = 5.0
        A_constrained_reference = array([[4.0,-4.0,4.0],
                                         [-5.0,5.0,-5.0],
                                         [-5.0,5.0,-5.0]])
        
        A_constrained = constrainComponents3by3(A, A_max)
        self.assertTrue(numpy.all(A_constrained==A_constrained_reference))
        
        A_constrained = A.copy()
        constrainComponents3by3InPlace(A_constrained, A_max)
        self.assertTrue(numpy.all(A_constrained==A_constrained_reference))

class TestTensorBasics(unittest.TestCase):
    
    
    def test_fourthOrderIdentity(self):
        """Test linearAlgebra.fourthOrderIdentity
        
        The tensor dot with a random second order tensor should return
        the same tensor.     
        """
        for n in range(1,4):
            I = fourthOrderIdentity(n)
            A = rand(n,n)
            IA = tensordot(I,A)
            self.assertTrue(numpy.all(A==IA))
            
    def test_TensorOrderConversions(self):
        """Test the fourth order <-> second order conversion functions.
        
        The functions covered are:
        - linearAlgebra.fourthOrderAsSecondOrder
        - linearAlgebra.fourthOrderAsSecondOrder3
        - linearAlgebra.secondOrderAsFourthOrder
        - linearAlgebra.secondOrderAsFourthOrder3
        
        Call the mapping from fourth order to second order \f$\mathcal{L}\f$.
        This creates two random fourth order tensors and checks that
        \f[
        A + B = \mathcal{L}^{-1} (\mathcal{L}(A) + \mathcal{L}(B))
        \f]
        
        """
        # Generic functions
        for n in range(1,4):
            A = rand(n,n,n,n)
            B = rand(n,n,n,n)
            A_plus_B_direct = A+B
            A2ndOrder_plus_B2ndOrder = fourthOrderAsSecondOrder(A) + \
                                       fourthOrderAsSecondOrder(B)              
            A_plus_B_converted = secondOrderAsFourthOrder(A2ndOrder_plus_B2ndOrder)
            self.assertTrue(numpy.all(A_plus_B_direct==A_plus_B_converted))
            
        # Optimised functions for specific dimensions
        for n in [3]:
            A = rand(n,n,n,n)
            B = rand(n,n,n,n)
            A_plus_B_direct = A+B
            A2ndOrder_plus_B2ndOrder = fourthOrderAsSecondOrder3(A) + \
                                       fourthOrderAsSecondOrder3(B)              
            A_plus_B_converted = secondOrderAsFourthOrder3(A2ndOrder_plus_B2ndOrder)
            self.assertTrue(numpy.all(A_plus_B_direct==A_plus_B_converted))
    
    def test_TensorProducts(self):
        """Test the tensor product functions.
        
        The functions covered are:
        - linearAlgebra.secondOrderOuterProduct
        - linearAlgebra.secondOrderOuterProduct3
        - linearAlgebra.tensorInner2_2
        - linearAlgebra.tensorInnerKalidindi2_2
        - linearAlgebra.tensordot4_2
        - linearAlgebra.tensordotKalidindi4_2
        """
        
        # 2nd order outer product, generic
        A = array([[1,2],[3,4]])
        B = array([[4,3],[2,1]])
        AB = secondOrderOuterProduct(A,B)
        AB_correct = array([
            [
                [
                    [4,3],   #A[0,0]*B[0,:]
                    [2,1]    #A[0,0]*B[1,:]
                ],
                [
                    [8,6],   #A[0,1]*B[0,:]
                    [4,2]    #A[0,1]*B[1,:]
                ]
            ],
            [
                [
                    [12,9],  #A[1,0]*B[0,:]
                    [6,3]    #A[1,0]*B[1,:]
                ],
                [
                    [16,12], #A[1,1]*B[0,:]
                    [8,4]    #A[1,1]*B[1,:]                
                ]
            ]        
        ])
        self.assertTrue(numpy.all(AB==AB_correct))
        
        # 2nd order outer product, optimised for 3d
        A = rand(3,3)
        B = rand(3,3)
        AB = secondOrderOuterProduct3(A,B)
        AB_reference = secondOrderOuterProduct(A,B)
        self.assertTrue(numpy.all(AB==AB_reference))
        
        # 2nd order double inner product: A_ij B_ji version
        A = rand(3,3)
        B = rand(3,3)
        A_ddot_B = tensorInner2_2(A,B)
        A_ddot_B_reference = einsum('ij,ji',A,B)
        self.assertTrue(abs(A_ddot_B-A_ddot_B_reference)/abs(A_ddot_B_reference) < MACHINE_PRECISION)
        
        # 2nd order double inner product: A_ij B_ij version
        A = rand(3,3)
        B = rand(3,3)
        A_ddot_B = tensorInnerKalidindi2_2(A,B)
        A_ddot_B_reference = einsum('ij,ij',A,B)
        self.assertTrue(abs(A_ddot_B-A_ddot_B_reference)/abs(A_ddot_B_reference) < MACHINE_PRECISION)
        
        # 4th order product with 2nd order: A_ijkl B_lk version
        A = rand(3,3,3,3)
        B = rand(3,3)
        AB = tensordot4_2(A,B)
        AB_reference = einsum('ijkl,lk->ij',A,B)
        self.assertTrue(norm(AB-AB_reference)/norm(AB_reference) < MACHINE_PRECISION)
        
        # 4th order product with 2nd order: A_ijkl B_kl version
        A = rand(3,3,3,3)
        B = rand(3,3)
        AB = tensordotKalidindi4_2(A,B)
        AB_reference = einsum('ijkl,kl->ij',A,B)
        self.assertTrue(norm(AB-AB_reference)/norm(AB_reference) < MACHINE_PRECISION)
    
    def test_TensorInverses(self):
        """Test the tensor inverse functions.
        
        The functions covered are:
        - linearAlgebra.fourthOrderInverse
        - linearAlgebra.fourthOrderInverse3
        """
        # Generic function
        for n in range(1,4):
            T = rand(n,n,n,n)
            T_inv = fourthOrderInverse(T)
            A = rand(n,n)
            
            # Condition number of T matrix for precision scaling
            T_cond = cond(fourthOrderAsSecondOrder(T))
            
            # Left inverse
            TA = tensordot(T,A)
            T_inv_TA = tensordot(T_inv, TA)
            self.assertTrue(norm(T_inv_TA-A)/T_cond < MACHINE_PRECISION)
            
            # Right inverse
            T_inv_A = tensordot(T_inv,A)
            T_T_inv_A = tensordot(T, T_inv_A)
            self.assertTrue(norm(T_T_inv_A-A)/T_cond < MACHINE_PRECISION)
            
        # Optimised functions for specific dimensions
        for n in [3]:
            T = rand(n,n,n,n)
            T_inv = fourthOrderInverse3(T)
            A = rand(n,n)
            
            # Condition number of T matrix for precision scaling
            T_cond = cond(fourthOrderAsSecondOrder3(T))
            
            # Left inverse
            TA = tensordotKalidindi4_2(T,A)
            T_inv_TA = tensordotKalidindi4_2(T_inv, TA)
            self.assertTrue(norm(T_inv_TA-A)/T_cond < MACHINE_PRECISION)
            
            # Right inverse
            T_inv_A = tensordotKalidindi4_2(T_inv,A)
            T_T_inv_A = tensordotKalidindi4_2(T, T_inv_A)
            self.assertTrue(norm(T_T_inv_A-A)/T_cond < MACHINE_PRECISION)
        
if __name__ == '__main__':
    unittest.main()
