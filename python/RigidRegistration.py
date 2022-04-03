from builtins import super
import numpy as np
import numbers
from pycpd.emregistration import EMRegistration
from pycpd.utility import is_positive_semi_definite


class RigidRegistrationnoScale(EMRegistration):
    """
    Rigid registration without scale.

    Attributes
    ----------
    R: numpy array (semi-positive definite)
        DxD rotation matrix. Any well behaved matrix will do,
        since the next estimate is a rotation matrix.

    t: numpy array
        1xD initial translation vector.

    s: float (positive)
        scaling parameter.

    A: numpy array
        Utility array used to calculate the rotation matrix.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

    YPY: float
        Denominator value used to update the scale factor.
        Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    X_hat: numpy array
        Centered target point cloud.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

    """

    def __init__(self, R=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if R is not None and (R.ndim is not 2 or R.shape[0] is not self.D or R.shape[1] is not self.D or not is_positive_semi_definite(R)):
            raise ValueError(
                'The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(self.D, self.D, R))

        if t is not None and (t.ndim is not 2 or t.shape[0] is not 1 or t.shape[1] is not self.D):
            raise ValueError(
                'The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(self.D, t))


        self.R = np.eye(self.D) if R is None else R
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # target point cloud mean
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0),
                        self.Np)
        # source point cloud mean
        muY = np.divide(
            np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.X_hat = self.X - np.tile(muX, (self.N, 1))
        # centered source point cloud
        Y_hat = self.Y - np.tile(muY, (self.M, 1))
        self.YPY = np.dot(np.transpose(self.P1), np.sum(
            np.multiply(Y_hat, Y_hat), axis=1))

        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

        # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D, ))
        C[self.D-1] = np.linalg.det(np.dot(U, V))

        # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        self.R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))

        self.t = np.transpose(muX) - np.dot(np.transpose(self.R), np.transpose(muY))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the rigid transformation.

        """
        if Y is None:
            self.TY = np.dot(self.Y, self.R) + self.t
            return
        else:
            return np.dot(Y, self.R) + self.t

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the rigid transformation.
        See the update rule for sigma2 in Fig. 2 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.q

        trAR = np.trace(np.dot(self.A, self.R))
        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.X_hat, self.X_hat), axis=1))
        self.q = (xPx - 2 * trAR + self.YPY) / \
            (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
        self.diff = np.abs(self.q - qprev)
        self.sigma2 = (xPx - trAR) / (self.Np * self.D)
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the rigid transformation parameters.

        """
        return self.R, self.t
