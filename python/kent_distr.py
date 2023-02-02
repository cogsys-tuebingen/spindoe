#!/usr/bin/env python

# Taken frrom: https://github.com/edfraenkel/kent_distribution/blob/master/kent_distribution.py
# Wiki page: https://en.wikipedia.org/wiki/Kent_distribution


"""
The algorithms here are partially based on methods described in:
[The Fisher-Bingham Distribution on the Sphere, John T. Kent
Journal of the Royal Statistical Society. Series B (Methodological)
Vol. 44, No. 1 (1982), pp. 71-80 Published by: Wiley
Article Stable URL: http://www.jstor.org/stable/2984712]
The example code in kent_example.py serves not only as an example
but also as a test. It performs some higher level tests but it also 
generates example plots if called directly from the shell.
"""

from numpy import *
from scipy.optimize import fmin_bfgs
from scipy.special import gamma as gamma_fun
from scipy.special import iv as modified_bessel_2ndkind
from scipy.special import ivp as modified_bessel_2ndkind_derivative
from scipy.stats import uniform

# to avoid confusion with the norm of a vector we give the normal distribution a less confusing name here
from scipy.stats import norm as gauss
from scipy.linalg import eig
import sys
import warnings


# helper function
def MMul(A, B):
    return inner(A, transpose(B))


# helper function to compute the L2 norm. scipy.linalg.norm is not used because this function does not allow to choose an axis
def norm(x, axis=None):
    if isinstance(x, list) or isinstance(x, tuple):
        x = array(x)
    return sqrt(sum(x * x, axis=axis))


def kent(theta, phi, psi, kappa, beta):
    """
    Generates the Kent distribution based on the spherical coordinates theta, phi, psi
    with the concentration parameter kappa and the ovalness beta
    """
    gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(
        theta, phi, psi
    )
    k = KentDistribution(gamma1, gamma2, gamma3, kappa, beta)
    return k


def kent2(gamma1, gamma2, gamma3, kappa, beta):
    """
    Generates the Kent distribution using the orthonormal vectors gamma1,
    gamma2 and gamma3, with the concentration parameter kappa and the ovalness beta
    """
    assert abs(inner(gamma1, gamma2)) < 1e-10
    assert abs(inner(gamma2, gamma3)) < 1e-10
    assert abs(inner(gamma3, gamma1)) < 1e-10
    return KentDistribution(gamma1, gamma2, gamma3, kappa, beta)


def kent3(A, B):
    """
    Generates the Kent distribution using the orthogonal vectors A and B
    where A = gamma1*kappa and B = gamma2*beta (gamma3 is inferred)
    A may have not have length zero but may be arbitrarily close to zero
    B may have length zero however. If so, then an arbitrary value for gamma2
    (orthogonal to gamma1) is chosen
    """
    kappa = norm(A)
    beta = norm(B)
    gamma1 = A / kappa
    if beta == 0.0:
        gamma2 = __generate_arbitrary_orthogonal_unit_vector(gamma1)
    else:
        gamma2 = B / beta
    theta, phi, psi = KentDistribution.gammas_to_spherical_coordinates(gamma1, gamma2)
    gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(
        theta, phi, psi
    )
    return KentDistribution(gamma1, gamma2, gamma3, kappa, beta)


def kent4(Gamma, kappa, beta):
    """
    Generates the kent distribution
    """
    gamma1 = Gamma[:, 0]
    gamma2 = Gamma[:, 1]
    gamma3 = Gamma[:, 2]
    return kent2(gamma1, gamma2, gamma3, kappa, beta)


def __generate_arbitrary_orthogonal_unit_vector(x):
    v1 = cross(x, array([1.0, 0.0, 0.0]))
    v2 = cross(x, array([0.0, 1.0, 0.0]))
    v3 = cross(x, array([0.0, 0.0, 1.0]))
    v1n = norm(v1)
    v2n = norm(v2)
    v3n = norm(v3)
    v = [v1, v2, v3][argmax([v1n, v2n, v3n])]
    return v / norm(v)


class KentDistribution(object):
    minimum_value_for_kappa = 1e-6

    @staticmethod
    def create_matrix_H(theta, phi):
        return array(
            [
                [cos(theta), -sin(theta), 0.0],
                [sin(theta) * cos(phi), cos(theta) * cos(phi), -sin(phi)],
                [sin(theta) * sin(phi), cos(theta) * sin(phi), cos(phi)],
            ]
        )

    @staticmethod
    def create_matrix_Ht(theta, phi):
        return transpose(KentDistribution.create_matrix_H(theta, phi))

    @staticmethod
    def create_matrix_K(psi):
        return array(
            [[1.0, 0.0, 0.0], [0.0, cos(psi), -sin(psi)], [0.0, sin(psi), cos(psi)]]
        )

    @staticmethod
    def create_matrix_Kt(psi):
        return transpose(KentDistribution.create_matrix_K(psi))

    @staticmethod
    def create_matrix_Gamma(theta, phi, psi):
        H = KentDistribution.create_matrix_H(theta, phi)
        K = KentDistribution.create_matrix_K(psi)
        return MMul(H, K)

    @staticmethod
    def create_matrix_Gammat(theta, phi, psi):
        return transpose(KentDistribution.create_matrix_Gamma(theta, phi, psi))

    @staticmethod
    def spherical_coordinates_to_gammas(theta, phi, psi):
        Gamma = KentDistribution.create_matrix_Gamma(theta, phi, psi)
        gamma1 = Gamma[:, 0]
        gamma2 = Gamma[:, 1]
        gamma3 = Gamma[:, 2]
        return gamma1, gamma2, gamma3

    @staticmethod
    def gamma1_to_spherical_coordinates(gamma1):
        theta = arccos(gamma1[0])
        phi = arctan2(gamma1[2], gamma1[1])
        return theta, phi

    @staticmethod
    def gammas_to_spherical_coordinates(gamma1, gamma2):
        theta, phi = KentDistribution.gamma1_to_spherical_coordinates(gamma1)
        Ht = KentDistribution.create_matrix_Ht(theta, phi)
        u = MMul(Ht, reshape(gamma2, (3, 1)))
        psi = arctan2(u[2][0], u[1][0])
        return theta, phi, psi

    def __init__(self, gamma1, gamma2, gamma3, kappa, beta):
        self.gamma1 = array(gamma1, dtype=float64)
        self.gamma2 = array(gamma2, dtype=float64)
        self.gamma3 = array(gamma3, dtype=float64)
        self.kappa = float(kappa)
        self.beta = float(beta)

        (
            self.theta,
            self.phi,
            self.psi,
        ) = KentDistribution.gammas_to_spherical_coordinates(self.gamma1, self.gamma2)

        for gamma in gamma1, gamma2, gamma3:
            assert len(gamma) == 3

        self._cached_rvs = array([], dtype=float64)
        self._cached_rvs.shape = (0, 3)

    @property
    def Gamma(self):
        return self.create_matrix_Gamma(self.theta, self.phi, self.psi)

    def normalize(self, cache=dict(), return_num_iterations=False):
        """
        Returns the normalization constant of the Kent distribution.
        The proportional error may be expected not to be greater than
        1E-11.


        >>> gamma1 = array([1.0, 0.0, 0.0])
        >>> gamma2 = array([0.0, 1.0, 0.0])
        >>> gamma3 = array([0.0, 0.0, 1.0])
        >>> tiny = KentDistribution.minimum_value_for_kappa
        >>> abs(kent2(gamma1, gamma2, gamma3, tiny, 0.0).normalize() - 4*pi) < 4*pi*1E-12
        True
        >>> for kappa in [0.01, 0.1, 0.2, 0.5, 2, 4, 8, 16]:
        ...     print abs(kent2(gamma1, gamma2, gamma3, kappa, 0.0).normalize() - 4*pi*sinh(kappa)/kappa) < 1E-15*4*pi*sinh(kappa)/kappa,
        ...
        True True True True True True True True
        """
        k, b = self.kappa, self.beta
        if not (k, b) in cache:
            G = gamma_fun
            I = modified_bessel_2ndkind
            result = 0.0
            j = 0
            if b == 0.0:
                result = ((0.5 * k) ** (-2 * j - 0.5)) * (I(2 * j + 0.5, k))
                result /= G(j + 1)
                result *= G(j + 0.5)

            else:
                while True:
                    a = exp(log(b) * 2 * j + log(0.5 * k) * (-2 * j - 0.5)) * I(
                        2 * j + 0.5, k
                    )
                    a /= G(j + 1)
                    a *= G(j + 0.5)
                    result += a

                    j += 1
                    if abs(a) < abs(result) * 1e-12 and j > 5:
                        break

            cache[k, b] = 2 * pi * result
        if return_num_iterations:
            return cache[k, b], j
        else:
            return cache[k, b]

    def log_normalize(self, return_num_iterations=False):
        """
        Returns the logarithm of the normalization constant.
        """
        if return_num_iterations:
            normalize, num_iter = self.normalize(return_num_iterations=True)
            return log(normalize), num_iter
        else:
            return log(self.normalize())

    def pdf_max(self, normalize=True):
        return exp(self.log_pdf_max(normalize))

    def log_pdf_max(self, normalize=True):
        """
        Returns the maximum value of the log(pdf)
        """
        if self.beta == 0.0:
            x = 1
        else:
            x = self.kappa * 1.0 / (2 * self.beta)
        if x > 1.0:
            x = 1
        fmax = self.kappa * x + self.beta * (1 - x**2)
        if normalize:
            return fmax - self.log_normalize()
        else:
            return fmax

    def pdf(self, xs, normalize=True):
        """
        Returns the pdf of the kent distribution for 3D vectors that
        are stored in xs which must be an array of N x 3 or N x M x 3
        N x M x P x 3 etc.

        The code below shows how points in the pdf can be evaluated. An integral is
        calculated using random points on the sphere to determine wether the pdf is
        properly normalized.

        >>> from numpy.random import seed
        >>> from scipy.stats import norm as gauss
        >>> seed(666)
        >>> num_samples = 400000
        >>> xs = gauss(0, 1).rvs((num_samples, 3))
        >>> xs = divide(xs, reshape(norm(xs, 1), (num_samples, 1)))
        >>> assert abs(4*pi*average(kent(1.0, 1.0, 1.0, 4.0,  2.0).pdf(xs)) - 1.0) < 0.01
        >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 4.0,  2.0).pdf(xs)) - 1.0) < 0.01
        >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 4.0,  8.0).pdf(xs)) - 1.0) < 0.01
        >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 16.0, 8.0).pdf(xs)) - 1.0) < 0.01
        """
        return exp(self.log_pdf(xs, normalize))

    def log_pdf(self, xs, normalize=True):
        """
        Returns the log(pdf) of the kent distribution.
        """
        axis = len(shape(xs)) - 1
        g1x = sum(self.gamma1 * xs, axis)
        g2x = sum(self.gamma2 * xs, axis)
        g3x = sum(self.gamma3 * xs, axis)
        k, b = self.kappa, self.beta

        f = k * g1x + b * (g2x**2 - g3x**2)
        if normalize:
            return f - self.log_normalize()
        else:
            return f

    def pdf_prime(self, xs, normalize=True):
        """
        Returns the derivative of the pdf with respect to kappa and beta.
        """
        return self.pdf(xs, normalize) * self.log_pdf_prime(xs, normalize)

    def log_pdf_prime(self, xs, normalize=True):
        """
        Returns the derivative of the log(pdf) with respect to kappa and beta.
        """
        axis = len(shape(xs)) - 1
        g1x = sum(self.gamma1 * xs, axis)
        g2x = sum(self.gamma2 * xs, axis)
        g3x = sum(self.gamma3 * xs, axis)
        k, b = self.kappa, self.beta

        dfdk = g1x
        dfdb = g2x**2 - g3x**2
        df = array([dfdk, dfdb])
        if normalize:
            return transpose(transpose(df) - self.log_normalize_prime())
        else:
            return df

    def normalize_prime(self, cache=dict(), return_num_iterations=False):
        """
        Returns the derivative of the normalization factor with respect to kappa and beta.
        """
        k, b = self.kappa, self.beta
        if not (k, b) in cache:
            G = gamma_fun
            I = modified_bessel_2ndkind
            dIdk = lambda v, z: modified_bessel_2ndkind_derivative(v, z, 1)
            dcdk, dcdb = 0.0, 0.0
            j = 0
            if b == 0:
                dcdk = (
                    (G(j + 0.5) / G(j + 1))
                    * ((-0.5 * j - 0.125) * (k) ** (-2 * j - 1.5))
                    * (I(2 * j + 0.5, k))
                )
                dcdk += (
                    (G(j + 0.5) / G(j + 1))
                    * ((0.5 * k) ** (-2 * j - 0.5))
                    * (dIdk(2 * j + 0.5, k))
                )

                dcdb = 0.0
            else:
                while True:
                    dk = (
                        (-1 * j - 0.25)
                        * exp(log(b) * 2 * j + log(0.5 * k) * (-2 * j - 1.5))
                        * I(2 * j + 0.5, k)
                    )
                    dk += exp(log(b) * 2 * j + log(0.5 * k) * (-2 * j - 0.5)) * dIdk(
                        2 * j + 0.5, k
                    )
                    dk /= G(j + 1)
                    dk *= G(j + 0.5)

                    db = (
                        2
                        * j
                        * exp(log(b) * (2 * j - 1) + log(0.5 * k) * (-2 * j - 0.5))
                        * I(2 * j + 0.5, k)
                    )
                    db /= G(j + 1)
                    db *= G(j + 0.5)
                    dcdk += dk
                    dcdb += db

                    j += 1
                    if (
                        abs(dk) < abs(dcdk) * 1e-12
                        and abs(db) < abs(dcdb) * 1e-12
                        and j > 5
                    ):
                        break

                # print "dc", dcdk, dcdb, "(", k, b

            cache[k, b] = 2 * pi * array([dcdk, dcdb])
        if return_num_iterations:
            return cache[k, b], j
        else:
            return cache[k, b]

    def log_normalize_prime(self, return_num_iterations=False):
        """
        Returns the derivative of the logarithm of the normalization factor.
        """
        if return_num_iterations:
            normalize_prime, num_iter = self.normalize_prime(return_num_iterations=True)
            return normalize_prime / self.normalize(), num_iter
        else:
            return self.normalize_prime() / self.normalize()

    def log_likelihood(self, xs):
        """
        Returns the log likelihood for xs.
        """
        retval = self.log_pdf(xs)
        return sum(retval, len(shape(retval)) - 1)

    def log_likelihood_prime(self, xs):
        """
        Returns the derivative with respect to kappa and beta of the log likelihood for xs.
        """
        retval = self.log_pdf_prime(xs)
        if len(shape(retval)) == 1:
            return retval
        else:
            return sum(retval, len(shape(retval)) - 1)

    def _rvs_helper(self):
        num_samples = 10000
        xs = gauss(0, 1).rvs((num_samples, 3))
        xs = divide(xs, reshape(norm(xs, 1), (num_samples, 1)))
        pvalues = self.pdf(xs, normalize=False)
        fmax = self.pdf_max(normalize=False)
        return xs[uniform(0, fmax).rvs(num_samples) < pvalues]

    def rvs(self, n_samples=None):
        """
        Returns random samples from the Kent distribution by rejection sampling.
        May become inefficient for large kappas.
        The returned random samples are 3D unit vectors.
        If n_samples == None then a single sample x is returned with shape (3,)
        If n_samples is an integer value N then N samples are returned in an array with shape (N, 3)
        """
        num_samples = 1 if n_samples == None else n_samples
        rvs = self._cached_rvs
        while len(rvs) < num_samples:
            new_rvs = self._rvs_helper()
            rvs = concatenate([rvs, new_rvs])
        if n_samples == None:
            self._cached_rvs = rvs[1:]
            return rvs[0]
        else:
            self._cached_rvs = rvs[num_samples:]
            retval = rvs[:num_samples]
            return retval

    def __repr__(self):
        return "kent(%s, %s, %s, %s, %s)" % (
            self.theta,
            self.phi,
            self.psi,
            self.kappa,
            self.beta,
        )
