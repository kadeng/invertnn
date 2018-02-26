'''
This module implements efficiently parameterized & quickly invertible Orthogonal Linear Transformations
based on a product of Householder Matrices (or Reflectors)

Additionally, efficiently parameterized invertible transformations based on Diagonal Matrices are
also provided in order to enable SVD Parametrization of arbitrary linear transformations

Based on:
Mhammedi: Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections
see https://arxiv.org/pdf/1612.00188.pdf

Zhang: Stabilizing Gradients for Deep Neural Networks via Efficient SVD Parameterization
see https://arxiv.org/pdf/1803.09327.pdf

Efficient Computation of the product of a Householder matrix and a vector (or another matrix)
see http://webhome.csc.uvic.ca/~dolesky/csc449-540/3.2b.pdf

:Author: Kai Londenberg, 2018
'''
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .invertible_transforms import InvertibleModule, InvertibleVolumePreservingMixin, InvertibleSequential

def simplified_hh_vector_product(u,a):
    '''
    Apply Householder Transformation to Vector or Matrix, given reflection Vector
    Args:
        u (torch.Tensor): Householder Reflection Vector of shape (n)
        a (torch.Tensor): Vector of Shape (n) or Matrix of shape (n, *)
    Returns:
        Transformed Vector or Matrix of same shape as a
    '''
    if len(a.shape)==1:
        a = a.unsqueeze(-1)
    n2 = torch.dot(u,u).sqrt()
    if (n2.item()==0.0):
        return a
    un = u / n2
    return a - (2.0 * un.unsqueeze(0).mm(a)) * un.unsqueeze(-1)


def create_hh_matrix(v):
    ''' 
    Create Householder Matrix from Reflection Vector
    :param v: Reflection Vector of shape (n)
    :return: Householder Matrix of shape (n,n)
    '''
    H = torch.eye(v.shape[0])
    n2sq = torch.dot(v, v)
    if n2sq==0.0:
       return H
    H -= (2 / n2sq) * torch.mm(v.unsqueeze(-1), v.unsqueeze(0))
    return H

def householder_qr(A):
    '''
    Perform QR decomposition based on a constructive inverse of Proof A.1 of Paper
    Zhang: Stabilizing Gradients for Deep Neural Networks via Efficient SVD Parameterization
    see https://arxiv.org/pdf/1803.09327.pdf

    This algorithm ensures that the resulting R has a nonnegative diagonal,
    and we get the proper reflection vectors.

    Args:
        A (torch.Tensor): Matrix to factorize
    Returns:
        Q (torch.Tensor): Orthogonal Matrix Q
        R (torch.Tensor): Upper triangular matrix with positive diagonal
        U (torch.Tensor): Upper Triangular Matrix of Reflection Vectors
    '''

    def _make_householder(a):
        v = a.clone() #- np.linalg.norm(a)(a[0] + np.copysign(np.linalg.norm(a), a[0]))
        v[0] -= a.norm(2)
        return v

    m, n = A.shape
    Q = torch.eye(m)
    U = torch.eye(m)
    for i in range(n - (m == n)):
        U[i,i:] = _make_householder(A[i:, i])
        H = create_hh_matrix(U[i,:])
        Q = torch.mm(Q, H)
        A = simplified_hh_vector_product(U[i,:], A)

    # Exception for n=1 as in proof
    # except that we apply it to the last index, instead of the first.
    if m==n:
        i=n-1
        if A[i,i]>0.0:
            U[i,i] = 0.0
        else:
            U[i,i] = 1.0
        H = create_hh_matrix(U[i,:])
        Q = torch.mm(Q, H)
        R = torch.mm(H, A)
    else:
        R = A
    return Q, R, U


class OrthogonalTransform(InvertibleVolumePreservingMixin, InvertibleModule, nn.Module):
    '''
    This module implements an efficiently parameterized & quickly invertible Orthogonal Linear Transformation
    based on a product of Householder Matrices (or Reflectors)

    The transform is equivalent to multiplication with an Orthogonal (square( Matrix
    (accessible as commputed property W of this module ) and optional subsequent addition of a bias term.

    The image of the product of these Reflectors is the entire Manifold of Orthogonal Matrices,
    so that efficient backprop is possible while maintaining the Orthogonality Property of the Transform.
    '''

    weight: nn.Parameter
    bias: nn.Parameter
    n: int
    m: int
    h1: float

    def __init__(self, n: int, m: int = None, h1: float = 1.0, bias=True):
        """
        Orthogonal Linear Transformation, efficiently parameterized & invertible.

        :param n: Dimensionality of Transform ( resulting Matrix W will have dimensions n x n )
        :param m: How many Householder Reflectors to use ( this trades off expressiveness VS speed ) - Defaults to n ( in which case we have full expressiveness )
        :param h1: Constant for lowest right coefficient of last householder reflection matrix, has to be .0 or 1.0. Only used if n==m
        :param bias: Whether to add / subtract a bias term. Defaults to true
        """
        super().__init__()
        if m is None:
            m = n
        assert (abs(h1) == 1.0)
        self.n = n
        self.m = m

        self.h1 = h1
        # Full parameter matrix, which needs to be constrained further to a lower triangular form (see property U below)
        self.weight = nn.Parameter(torch.randn((n, m)))
        if bias:
            self.bias = nn.Parameter(torch.randn((1, n)))
        else:
            self.register_parameter('bias', None)

    @property
    def U(self):
        '''
        Constrained parameter Matrix U.

        This is weight constrained to lower triangular, with U[-1,-1] set to h1 if n==m
        see Mhammedi: Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections
        '''
        U = self.weight.triu()
        if self.n == self.m:
            U.data[-1, -1] = self.h1
        return U



    @property
    def W(self):
        '''
        Orthogonal Matrix W, computed on-the-fly, representing this transformation without bias term.
        '''
        return self._forward(torch.eye(self.n, requires_grad=False))

    @W.setter
    def W(self, new_W):
        assert(torch.slogdet(new_W)[1].abs().item()<1e-5) # Simple check for orthogonality
        Q, R, U = householder_qr(new_W.t())
        self.weight.data.copy_(U)
        self.h1 = U[-1,-1].item()

    def _forward(self, x):
        h = x.t().contiguous()
        U = self.U
        #Un = U.div((U * U).sum(dim=0).unsqueeze(0))
        for i in reversed(range(self.m)):
            h = simplified_hh_vector_product(U[i, :], h)

        return h.t().contiguous()

    def forward(self, x):
        return self._forward(x)

    def invert(self, x):
        '''
        Peforms inverted Orthogonal Transform of x
        equivalent to self.W.t().matmul(x), but slightly more efficient

        Args:
            x ( torch.Tensor): Input tensor to transform
        Returns:
            Transformed tensor of same shape as x
        '''
        h = x.t().contiguous()
        U = self.U
        for i in range(self.m):
            h = simplified_hh_vector_product(U[i, :], h)
        return h.t().contiguous()

class OrthogonalTransform2D(OrthogonalTransform):
    '''
    Pixelwise Orthogonal Transformation
    see :class:`.OrthogonalTransform`
    '''

    def forward(self, x):
        '''
        Peforms forward Orthogonal Transform of x as a pixelwise 2D Convolution

        Args:
            x (torch.Tensor): Tensor to transform
        Returns:
            Transformed tensor of same shape as x
        '''
        W = self.W.view(self.n, self.n, 1, 1)
        if self.bias is not None:
            bias = self.bias.squeeze()
        else:
            bias = None
        return F.conv2d(x, W, bias, (1, 1), (0, 0), (1, 1), 1)

    def invert(self, x):
        '''
        Peforms inverted Orthogonal Transform of x
        equivalent to self.W.t().matmul(x), but slightly more efficient
        Args:
            x (torch.Tensor): Input Tensor to be transformed
        Returns:
            Transformed tensor of same shape as x
        '''
        Wt = self.W.t().view(self.n, self.n, 1, 1)
        if self.bias is not None:
            bias = self.bias.view(1, self.bias.shape[1], 1, 1).contiguous()
            return F.conv2d(x - bias, Wt, None, (1, 1), (0, 0), (1, 1), 1)
        else:
            return F.conv2d(x, Wt, None, (1, 1), (0, 0), (1, 1), 1)


class DiagonalLinearTransform(InvertibleModule, nn.Module):
    '''
    This module implements an invertible transformation equivalent to a multiplication with a Diagonal Matrix

    It can be combined with Orthogonal Transformations to reparametrize arbitrary linear transformations
    '''

    weight: nn.Parameter
    bias: nn.Parameter
    n: int

    def __init__(self, n: int, bias=True):
        """
        Orthogonal Diagonal Transformation, efficiently parameterized & invertible.

        :param n: Dimensionality of Transform ( resulting Matrix W will have dimensions n x n )
        :param bias: Whether to add / subtract a bias term. Defaults to true
        """
        super().__init__()
        self.n = n
        # Full parameter matrix, which needs to be constrained further to a lower triangular form (see property U below)
        self.weight = nn.Parameter(torch.randn(1, n))
        if bias:
            self.bias = nn.Parameter(torch.randn(1, n))
        else:
            self.register_parameter('bias', None)

    @property
    def W(self):
        '''
        Diagonal Matrix W, computed on-the-fly, representing this transformation without bias term.
        '''
        return torch.diag(self.weight.squeeze())

    @property
    def W_inv(self):
        '''
        Diagonal Matrix W, computed on-the-fly, representing this transformation without bias term.
        '''
        return torch.diag(self.weight.reciprocal().squeeze())

    def _forward(self, x):
        if self.bias is not None:
            return x * self.weight + self.bias
        else:
            return x * self.weight

    def forward(self, x):
        '''
        Peforms Diagonal Transform of x
        equivalent to (1.0/self.W).matmul(x - bias), but more efficient
        Returns:
             Transformed Tensor of same shape as x
        '''
        return self._forward(x)

    def invert(self, x):
        '''
        Peforms inverted Diagonal Transform of x
        equivalent to (1.0/self.W).matmul(x - bias), but more efficient
        Returns:
             Transformed Tensor of same shape as x
        '''
        if self.bias is not None:
            return (x - self.bias) / self.weight
        else:
            return x / self.weight

    def inv_jacobian_logabsdet(self, x):
        r = self.invert(x)
        winv = self.weight.reciprocal().abs().log().sum()
        return r, winv.expand(x.shape[0], 1).contiguous()


class DiagonalLinearTransform2D(DiagonalLinearTransform):
    '''
    Pixelwise Diagonal Linear Transformation
    '''

    def forward(self, x):
        '''
        Peforms forward Orthogonal Transform of x as a pixelwise 2D Convolution
        '''
        W = self.W.view(self.n, self.n, 1, 1)
        if self.bias is not None:
            bias = self.bias.squeeze()
        else:
            bias = None
        return F.conv2d(x, W, bias, (1, 1), (0, 0), (1, 1), 1)

    def invert(self, x):
        '''
        Peforms inverted Orthogonal Transform of x
        equivalent to self.W.t().matmul(x), but slightly more efficient
        :return: Transformed x
        '''
        Wt = self.W_inv.view(self.n, self.n, 1, 1)
        if self.bias is not None:
            bias = self.bias.view(1, self.bias.shape[1], 1, 1).contiguous()
            return F.conv2d(x - bias, Wt, None, (1, 1), (0, 0), (1, 1), 1)
        else:
            return F.conv2d(x, Wt, None, (1, 1), (0, 0), (1, 1), 1)


class InvertibleLinear(InvertibleSequential):

    def __init__(self, n, m=None, bias=True):
        super().__init__(OrderedDict(
            [('U', OrthogonalTransform(n, m, bias=False)),
            ('S', DiagonalLinearTransform(n, bias=False)),
            ('V', OrthogonalTransform(n, m, bias=bias))]
        ))

class InvertibleLinear2D(InvertibleSequential):

    def __init__(self, n, m=None, bias=True):
        super().__init__(OrderedDict(
            [('U', OrthogonalTransform2D(n, m, bias=False)),
            ('S', DiagonalLinearTransform2D(n, bias=False)),
            ('V', OrthogonalTransform2D(n, m, bias=bias))]
        ))
