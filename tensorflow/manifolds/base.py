import tensorflow as tf
import numpy as np
from math import sqrt
from ..utils.functional import tanh, atanh_

class Manifold(object):
    """
    This class can be used for mathematical functions on the Poincare ball.
    The Poincare n-ball with curvature -c< 0 is the set of points n-dim Euclidean space such that |x|^2 < 1/c
    with a Riemannian metric given by 
    math::
        \lambda_c g_E = \frac{1}{1-c \|x\|^2} g_E
    where g_E is the standard Euclidean metric.
    """

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    @property
    def curvature(self):
        return self._c

    @curvature.setter
    def curvature(self, c):
        raise NotImplementedError

    def mobius_matvec(self, m, x):
        """
        Generalization for matrix-vector multiplication to hyperbolic space.
        Args:
            m : Tensor for multiplication
            x : Tensor point on poincare ball
        Returns
            Mobius matvec result
        """
        raise NotImplementedError

    def clipped_norm(self, x):
        raise NotImplementedError

    def expmap(self, u, p):
        raise NotImplementedError

    def lambda_x(self, x):
        raise NotImplementedError

    def expmap0(self, u):
        """
        Hyperbolic exponential map at zero.
          Args:
            u: tensor of size B x dimension representing tangent vectors.
          Returns:
            Tensor of shape B x dimension.
          """
        raise NotImplementedError

    def logmap0(self, p):
        """
        Hyperbolic logarithmic map at zero.
        Args:
          p: tensor of size B x dimension representing hyperbolic points.
        Returns:
          Tensor of shape B x dimension.
        """
        raise NotImplementedError

    def proj(self, x):
        """
        Safe projection on the manifold for numerical stability. This was mentioned in [1]

        Args:
            x : Tensor point on the Poincare ball
        Returns:
            Projected vector on the manifold

        References:
            [1] Hyperbolic Neural Networks, NIPS2018
            https://arxiv.org/abs/1805.09112
        """
        raise NotImplementedError

    def mobius_add(self, x, y):
        """Element-wise Mobius addition.
        Args:
        x: Tensor of size B x dimension representing hyperbolic points.
        y: Tensor of size B x dimension representing hyperbolic points.
        c: Tensor of size 1 representing the absolute hyperbolic curvature.
        Returns:
        Tensor of shape B x dimension representing the element-wise Mobius addition
        of x and y.
        """
        raise NotImplementedError
    
    def gyr(x, y, z):
        """
        Ungar's gryation operation defined in [1].
        Args:
            x, y, z: Tensors of size B x dim in the Poincare ball of curvature c
        Returns:
            Tensor of size B x dim
        Reference:
           [1] A. Ungar, A Gryovector Space Approach to Hyperbolic Geometry
        """
        raise NotImplementedError

    def parallel_transport(self, x, y, v):
        """
        The parallel transport of the tangent vector v from the tangent space at x
        to the tangent space at y
        """
        raise NotImplementedError

    def dist(self, x, y):
        """ Hyperbolic distance between points 
        Args:
            x, y: Tensors of size B x dim of points in the Poincare ball
        """
        raise NotImplementedError

    def reflect(self, a, x):
        """ Hyperbolic reflection with center at a, i.e. sphere inversion
        about the sphere centered at a orthogonal.
          Args:
            a: Tensor representing center of reflection 
            x: Tensor of size B x dim in the Poincare ball
        Returns:
            size B x dim Tensor of reflected points
        """
        raise NotImplementedError

    def reflect0(self, z, x):
        """ Hyperbolic reflection that maps z to 0 and 0 to z. 

        Args:
            z: point in the Poincare ball that maps to the origin 
            x: Tensor of size B x dim representing B points to reflect
        Returns:
            size B x dim Tensor of reflected points
        """
        raise NotImplementedError

