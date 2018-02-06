""" A library for linear algebra and tensor manipulations.
"""

import numpy
from numba.decorators import jit
from numpy import zeros, einsum, sqrt, tensordot, eye, ones
from numpy.linalg import inv, solve, det
from numpy.random import normal
from numpy import sign, outer, dot, trace

@jit
def fourthOrderIdentity(n=3):
    """Return the fourth order identity tensor \f$\mathcal{I}\f$ of dimension \f$n \times n \times n \times n\f$.
    
    The tensor is defined by
    \f[
    I_{ijkl} = \delta_{ik}\delta_{jl}
    \f]    
    
    Test coverage in linearAlgebra_test.TestTensorBasics.test_fourthOrderIdentity.
    
    Args:
        n: The dimension \f$ n \f$
    Returns:
        \f$\mathcal{I}\f$
    """
    I = zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    I[i,j,k,l] = (i==k)*(j==l)
    return I

@jit 
def fourthOrderAsSecondOrder(T):
    """Flatten a 4th order tensor into a matrix.
    
    The 4th order tensor \f$\mathbf{T}\f$, of dimension \f$n \times n \times n \times n\f$
    is flattened into the matrix \f$A\f$, of dimension \f$ n^2 \times n^2 \f$
    according to:
    \f[
    T_{ijkl} = A_{in+j,kn+l}.
    \f]
    That is, it's a row-major flattening.
    
    Test coverage in linearAlgebra_test.TestTensorBasics.test_TensorOrderConversions.
    
    Args:
        T: the 4th order tensor \f$\mathbf{T}\f$
    Returns:
        A: the matrix \f$A\f$
    """
    initial_shape = T.shape
    dim4 = initial_shape[0]
    for i in range(1,4):
        if initial_shape[i] != dim4:
            raise TypeError("Tensor is not square.")
    
    dim2 = dim4**2
    A = zeros((dim2,dim2))
    for i4 in range(dim4):
        for j4 in range(dim4):
            for k4 in range(dim4):
                for l4 in range(dim4):
                    i2 = i4*dim4+j4
                    j2 = k4*dim4+l4
                    A[i2,j2] = T[i4,j4,k4,l4]
                    
    return A
    
@jit 
def fourthOrderAsSecondOrder3(T):
    """Flatten a 3x3x3x3 tensor into a 9x9 matrix.
    
    This is a special case of linearAlgebra.fourthOrderAsSecondOrder that
    allows for better JIT compilation by Numba.
    
    Test coverage in linearAlgebra_test.TestTensorBasics.test_TensorOrderConversions.
    
    Args:
        T: the 3x3x3x3 tensor \f$\mathbf{T}\f$
    Returns:
        A: the 9x9 matrix \f$A\f$
    """
    A = zeros((9,9))
    for i4 in range(3):
        for j4 in range(3):
            for k4 in range(3):
                for l4 in range(3):
                    i2 = i4*3+j4
                    j2 = k4*3+l4
                    A[i2,j2] = T[i4,j4,k4,l4]
                    
    return A
    
@jit 
def secondOrderAsFourthOrder(A):
    """Pack a matrix into a 4th order tensor.
    
    This is the reverse of linearAlgebra.fourthOrderAsSecondOrder. See
    that documentation for the definitions.
    
    Test coverage in linearAlgebra_test.TestTensorBasics.test_TensorOrderConversions.
    
    Args:
        A: the matrix \f$A\f$
    Returns:
        The fourth order tensor \f$T\f$
    """
    initial_shape = A.shape
    dim2 = initial_shape[0]
    for i in range(1,2):
        if initial_shape[i] != dim2:
            raise TypeError("Tensor is not square.")
    if sqrt(dim2) != int(sqrt(dim2)):
        raise TypeError("Tensor dimensions are not square numbers.")
        
    dim4 = int(sqrt(dim2))
    T = zeros((dim4,dim4,dim4,dim4))
    for i4 in range(dim4):
        for j4 in range(dim4):
            for k4 in range(dim4):
                for l4 in range(dim4):
                    i2 = i4*dim4+j4
                    j2 = k4*dim4+l4
                    T[i4,j4,k4,l4] = A[i2,j2]
                    
    return T

@jit 
def secondOrderAsFourthOrder3(A):
    """Flatten a 3x3x3x3 tensor into a 9x9 matrix.
    
    This is a special case of linearAlgebra.secondOrderAsFourthOrder that
    allows for better JIT compilation by Numba.
    
    Test coverage in linearAlgebra_test.TestTensorBasics.test_TensorOrderConversions.
    
    Args:
        A: the 9x9 matrix \f$A\f$
    Returns:
        The 3x3x3x3 tensor \f$T\f$
    """
    T = zeros((3,3,3,3))
    for i4 in range(3):
        for j4 in range(3):
            for k4 in range(3):
                for l4 in range(3):
                    i2 = i4*3+j4
                    j2 = k4*3+l4
                    T[i4,j4,k4,l4] = A[i2,j2]
                    
    return T

@jit
def secondOrderOuterProduct(A,B):
    """Return the outer product of two second order tensors.
    
    The product is
    \f[
        \mathbf{C} = \mathbf{A} \otimes \mathbf{B}.
    \f]
    
    Test coverage in linearAlgebra_test.TestTensorBasics.test_TensorProducts.
    
    Args:
        A: \f$\mathbf{A}\f$
        B: \f$\mathbf{B}\f$
        
    Returns:
        \f$\mathbf{C}\f$
    """
    return einsum('ij,kl->ijkl',A,B)
    
@jit
def secondOrderOuterProduct3(A,B):
    """Return the outer product of two 3x3 tensors.
    
    This is a special case of linearAlgebra.secondOrderOuterProduct that
    allows for better JIT compilation by Numba. See that documentation
    for the definitions.
    
    Test coverage in linearAlgebra_test.TestTensorBasics.test_TensorProducts.
    
     Args:
        A: \f$\mathbf{A}\f$
        B: \f$\mathbf{B}\f$
        
    Returns:
        \f$\mathbf{C}\f$
    """
    C = zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j,k,l] = A[i,j]*B[k,l]
    return C

@jit 
def tensorInner2_2(A,B):
    """Return the inner product of two 3x3 tensors.
    
    This is the \f$C = A_{ij}\f$ \f$B_{ji}\f$ kind. Test coverage in 
    linearAlgebra_test.TestTensorBasics.test_TensorProducts.
    
    Args:
        A: \f$\mathbf{A}\f$
        B: \f$\mathbf{B}\f$
    Returns:
        The product \f$C\f$
    """
    C = 0
    for i in range(3):
        for j in range(3):
            C += A[i,j]*B[j,i]
    return C

@jit('f8(f8[:,:],f8[:,:])',nopython=True)
def tensorInnerKalidindi2_2(A,B):
    """Return the inner product of two 3x3 tensors.
    
    This is the \f$C = A_{ij}\f$ \f$B_{ij}\f$ kind. Test coverage in 
    linearAlgebra_test.TestTensorBasics.test_TensorProducts.
    
    Args:
        A: \f$\mathbf{A}\f$
        B: \f$\mathbf{B}\f$
    Returns:
        The product \f$C\f$
    """
    C = 0
    for i in range(3):
        for j in range(3):
            C += A[i,j]*B[i,j]
    return C

@jit 
def tensordot4_2(A,B):
    """Return the inner product of a 3x3x3x3 tensor and a 3x3 tensor.
    
    This is the \f$C_{ij} = A_{ijkl}\f$ \f$B_{lk}\f$ kind. Test coverage in 
    linearAlgebra_test.TestTensorBasics.test_TensorProducts.
    
    Args:
        A: \f$\mathbf{A}\f$
        B: \f$\mathbf{B}\f$
    Returns:
        The product \f$\mathbf{C}\f$
    """
    C = zeros((3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j] += A[i,j,k,l]*B[l,k]
    return C
    
@jit('f8[:,:](f8[:,:,:,:],f8[:,:])') 
def tensordotKalidindi4_2(A,B):
    """Return the inner product of a 3x3x3x3 tensor and a 3x3 tensor.
    
    This is the \f$C_{ij} = A_{ijkl}\f$ \f$B_{kl}\f$ kind. Test coverage in 
    linearAlgebra_test.TestTensorBasics.test_TensorProducts.
    
    Args:
        A: \f$\mathbf{A}\f$
        B: \f$\mathbf{B}\f$
    Returns:
        The product \f$\mathbf{C}\f$
    """
    C = zeros((3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j] += A[i,j,k,l]*B[k,l]
    return C
 
@jit('void(f8[:,:,:,:],f8[:,:],f8[:,:])')   
def tensordotKalidindi4_2_in_place(A,B,C):
    """Return the inner product of a 3x3x3x3 tensor and a 3x3 tensor.
    
    This is the \f$C_{ij} = A_{ijkl}\f$ \f$B_{kl}\f$ kind. Test coverage in 
    linearAlgebra_test.TestTensorBasics.test_TensorProducts.
    
    Args:
        A: \f$\mathbf{A}\f$
        B: \f$\mathbf{B}\f$
    Returns:
        The product \f$\mathbf{C}\f$
    """
    for i in range(3):
        for j in range(3):
            C[i,j] = 0.0
            for k in range(3):
                for l in range(3):
                    C[i,j] += A[i,j,k,l]*B[k,l]
    
@jit
def fourthOrderInverse(T):
    """Inverse of a 4th tensor with respect to the double contraction product.
    
    The inverse satisfies
    \f[
        \mathbf{T}:\mathbf{T}^{-1} = \mathbf{T}^{-1}:\mathbf{T} = \mathcal{I},
    \f]
    where \f$\mathcal{I}\f$ is as defined in linearAlgebra.fourthOrderIdentity, 
    and the double contraction \f$:\f$ is defined as in linearAlgebra.tensordotKalidindi4_2.
    
    Test coverage in linearAlgebra_test.TestTensorBasics.test_TensorInverses.
    
    Args:
        T: the tensor \f$\mathbf{T}\f$
    Returns:
        The inverse \f$\mathbf{T}^{-1}\f$
    """
    initial_shape = T.shape
    n = initial_shape[0]
    for i in range(1,4):
        if initial_shape[i] != n:
            raise TypeError("Tensor is not square.")
            
    A = fourthOrderAsSecondOrder(T)
    I = fourthOrderAsSecondOrder(fourthOrderIdentity(n))
    A_inv = inv(A).dot(I)
    T_inv = secondOrderAsFourthOrder(A_inv)
    return T_inv
   
@jit
def fourthOrderInverse3(T):
    """Inverse of a 3x3x3x3 tensor.
    
    This is a special case of linearAlgebra.fourthOrderInverse that
    allows for better JIT compilation by Numba. See that documentation
    for the definitions.
    
    Test coverage in linearAlgebra_test.TestTensorBasics.test_TensorInverses.
    
    Args:
        T: the tensor \f$\mathbf{T}\f$
    Returns:
        The inverse \f$\mathbf{T}^{-1}\f$
    """
    A = fourthOrderAsSecondOrder3(T)
    A_inv = solve(A, fourthOrderAsSecondOrder(fourthOrderIdentity(3)))
    T_inv = secondOrderAsFourthOrder3(A_inv)
    return T_inv
      
@jit
def constrainComponents3by3InPlace(A, A_max):
    """Constrain the magnitude of the components of a 3x3 array in place.
    
    Computes
    \f[
        A_{ij} \leftarrow  \mathrm{min}(|A_{ij}|, A_{\mathrm{max}}) \mathrm{sgn} (A_{ij})
    \f]
    in place, overwriting the entries of \f$\mathbf{A}\f$.
    
    Test coverage in linearAlgebra_test.TestArrayBasics.test_ConstrainComponents
    
    Args:
        A: \f$\mathbf{A}\f$
        A_max: \f$A_{\mathrm{max}}\f$
        
    Returns:
        None
    """
    for i in range(3):
        for j in range(3):
            A_abs = abs(A[i,j])
            if A_abs > A_max:
                A[i,j] = A[i,j]*A_max/A_abs
                
@jit
def constrainComponents3by3(A, A_max):
    """Constrain the magnitude of the components of a 3x3 array.
    
    Returns \f$\mathbf{A}^*\f$, where
    \f[
        A_{ij}^* =  \mathrm{min}(|A_{ij}|, A_{\mathrm{max}}) \mathrm{sgn} (A_{ij}).
    \f]
    
    Test coverage in linearAlgebra_test.TestArrayBasics.test_ConstrainComponents
    
    Args:
        A: \f$\mathbf{A}\f$
        A_max: \f$A_{\mathrm{max}}\f$
        
    Returns:
        \f$\mathbf{A}^*\f$
    """
    A_constrained = zeros((3,3))
    for i in range(3):
        for j in range(3):
            A_abs = abs(A[i,j])
            if A_abs > A_max:
                A_constrained[i,j] = A[i,j]*A_max/A_abs
            else:
                A_constrained[i,j] = A[i,j]
    return A_constrained

@jit('void(f8[:,:,:,:],f8[:,:,:,:])')
def fourthOrderIncrement3(A, B):
    """Increment a 3x3x3x3 tensor.
    
    Evaluates
    \f[
        A \leftarrow A + B
    \f]
    
    Args:
        A: \f$A\f$
        B: \f$B\f$
    Returns:
        None. \f$A\f$ is updated in place.
    """
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    A[i,j,k,l] += B[i,j,k,l]

@jit('f8[:,:](f8[:],f8[:])')
def numbaOuter3(v,w):
    """Computes the outer product of two vectors.
    
    Evaluates
    \f[
        \mathbf{A} = \mathbf{v} \otimes \mathbf{w}
    \f]
    This is a version for JIT compilation with Numba.
    
    Args:
        v: \f$\mathbf{v}\f$
        w: \f$\mathbf{w}\f$
    Returns:
        A: \f$\mathbf{A}\f$
    """
    A = zeros((3,3))
    for i in range(3):
        for j in range(3):
            A[i,j] = v[i]*w[j]
    return A
    
def deviatoricComponent3(A):
    """Return the deviatoric component of a 3x3 tensor.
    
    Defined by
    \f[
        A' = A_ij - \frac{A_{kk}}{3} \delta_{ij}
    \f]
    """
    return A - (trace(A)/3.0)*eye(3)

# This is taken verbatim from scipy/linalg/tests/test_decomp.py. 
def random_rot(dim):
    """Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(n)).
    The algorithm is described in the paper
    Stewart, G.W., 'The efficient generation of random orthogonal
    matrices with an application to condition estimators', SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization"""
    H = eye(dim)
    D = ones((dim,))
    for n in range(1, dim):
        x = normal(size=(dim-n+1,))
        D[n-1] = sign(x[0])
        x[0] -= D[n-1]*sqrt((x*x).sum())
        # Householder transformation

        Hx = eye(dim-n+1) - 2.*outer(x, x)/(x*x).sum()
        mat = eye(dim)
        mat[n-1:,n-1:] = Hx
        H = dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = -D.prod()
    H = (D*H.T).T
    return H
    
@jit('void(f8[:,:])')
def scaleToUnitDeterminant3(A):
    determinant = det(A)
    det_cube_root = determinant**(1.0/3)
    for i in range(3):
        for j in range(3):
            A[i,j] /= det_cube_root

@jit('f8[:,:](f8[:,:],f8[:,:])')
def matMult3(A,B):
    C = zeros((3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i,j] += A[i,k]*B[k,j]
    return C

@jit('f8[:,:](f8[:,:],f8[:,:])')
def AB_plusB_T_A3(A, B):
    """A*B + (B^T)*A
    """
    C = zeros((3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i,j] += A[i,k]*B[k,j] + B[k,i]*A[k,j]
    return C
    
@jit('void(f8[:,:],f8[:,:],f8[:,:])')
def AB_plusB_T_A3_in_place(A, B, C):
    """A*B + (B^T)*A
    """
    for i in range(3):
        for j in range(3):
            C[i,j] = 0.0
            for k in range(3):
                C[i,j] += A[i,k]*B[k,j] + B[k,i]*A[k,j]
                
def resampleFunction(x1, y1, x2):
    """Samples a discretely defined function over a new domain.
    
    Args:
        x1: the old domain
        y1: the old range
        x2: the new domain
    Returns:
        y2: the new range
    """
    l1 = len(x1)
    l2 = len(x2)
    y2 = [0.0*y1[0] for i in range(l2)]

    i1 = 0
    i2 = 0

    #Step x2 ahead of x1 to initialize
    while(x2[i2] <= x1[i1]):
        y2[i2] = y1[i1]
        i2 += 1

    while(i2 < l2 and i1 < l1-1):	
        #Step until x1 is ahead of x2	
        while(x1[i1] <= x2[i2] and i1 < l1-1):
            i1 += 1
        #Linearly interpolate values until x2 is ahead of x1
        while(x2[i2] <= x1[i1] and i2 < l2):
            y2[i2] = y1[i1-1] + (y1[i1]-y1[i1-1])*(x2[i2]-x1[i1-1])/(x1[i1]-x1[i1-1])
            i2 += 1
    
    return y2
    
def L2NormFunctionValues(x, y):
    norm = 0.0
    for i in range(len(x)-1):
        norm += (x[i+1]-x[i])*(y[i]**2)
    return sqrt(norm)
