"""
Spin regression with RANSAC based on QuateRA
We first use RANSAC to find the valid orientions ie that are on the same plane.
Then, the angle between the successive orientations are calculated and the spin norm is regressed from that
"""

import numpy as np
import time
from scipy.spatial.transform import Rotation as R


class SpinRegressor:
    def __init__(self, n=4, k=10, t=5e-2, d=3) -> None:

        # RANSAC parameters
        self.n = n  # Initial sample size for regression
        self.k = k  # Maximum number of iterations
        self.t = t  # Threshold value to determine if orientations fit well the model
        self.d = d  # Number of datapoints required to assert that the model is valid

    def get_spin_axis(self, y):
        # print("Y")
        # print(y)
        Z = y.T @ y
        # print("Z")
        # print(Z)
        u, s, vh = np.linalg.svd(Z, full_matrices=True)
        # print("Sigma")
        # print(s)
        u1 = self.rot_from_quat(u[:, 0])
        u2 = self.rot_from_quat(u[:, 1])
        axis = u2 * u1.inv()
        # print("u")
        # print(u)
        # print("U1: {}".format(u[:, 0]))
        # print("U2: {}".format(u[:, 1]))
        return axis.as_quat()[:3], u[:, 0], u[:, 1]

    def get_qtls(self, q, u1, u2):
        qtls = np.dot(q, u1) * u1 + np.dot(q, u2) * u2
        residual_error = 1 - np.sqrt(np.dot(q, u1) ** 2 + np.dot(q, u2) ** 2)
        qtls = qtls / np.sqrt(np.dot(q, u1) ** 2 + np.dot(q, u2) ** 2)
        return qtls, residual_error

    def get_phi(self, q, u1, u2):
        phis = 2 * np.arctan2(np.dot(u2, q), np.dot(u1, q))
        return phis

    def spin_norm_reg(self, phis, t):
        H = np.ones((len(phis[1:]), 2))
        dt = t[1:] - t[0]
        H[:, 1] = dt
        phi = np.array(phis[1:])
        X = np.linalg.inv(H.T @ H) @ H.T @ phi
        return X  # TODO: returns 2 values, must just return norm

    # THe scipy convention for quaternion is (x, y, z, w)
    # We use the (w, x , y, z) convention
    def rot_from_quat(self, q):
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        return rot

    def quat_from_rot(self, rot):
        q = rot.as_quat()
        return np.array([q[3], q[0], q[1], q[2]])

    def regress(self, t, rots):
        """
        Simplest regression method - implementation of QuateRA

        :param t np.array: Times [s] at which the rots where taken
        :param rots scipy.Rotation: Sequential ball orientation
        """
        assert len(t) == len(rots)
        n = len(t)
        y = np.zeros((n, 4))
        for i in range(n):
            y[i] = self.rot2quat(rots[i])
        axis, u1, u2 = self.get_spin_axis(y)
        # print(axis)
        qtls = np.zeros((n, 4))
        phis = np.zeros(n)
        # print(phis)
        # print(y)
        for i in range(n):
            q, _ = self.get_qtls(y[i], u1, u2)
            qtls[i, :] = q
            # print(q)
            phis[i] = self.get_phi(qtls[i], u1, u2)

        # print(phis)
        # Necessary to unwrap the angles for the linear regression
        phis = np.unwrap(phis)
        spin_norm = self.spin_norm_reg(phis, t)
        print(spin_norm)
        spin = -axis * spin_norm[1]  # HACK: minus sign had to be added

        first_rot = self.quat2rot(qtls[0])
        return spin, first_rot

    def RANSAC_regress(self, t, rots):
        assert len(t) == len(rots)
        n = len(t)
        y = np.zeros((n, 4))
        for i in range(n):
            y[i] = self.rot2quat(rots[i])
        axis, u, valid_idx, error = self.get_RANSAC_spin_axis(y)

        print()
        print(valid_idx)
        if axis is None:
            return None, None
        # Get the useful rotations
        n_sel = valid_idx.shape[0]
        y_sel = y[valid_idx]
        t_sel = t[valid_idx]
        u1, u2 = u[0], u[1]

        qtls = np.zeros((n_sel, 4))
        phis = np.zeros(n_sel)
        # print(phis)
        # print(y)
        for i in range(n_sel):
            q, _ = self.get_qtls(y_sel[i], u1, u2)
            qtls[i, :] = q
            # print(q)
            phis[i] = self.get_phi(qtls[i], u1, u2)

        # Necessary to unwrap the angles for the linear regression
        phis = np.unwrap(phis)
        spin_norm = self.spin_norm_reg(phis, t_sel)
        # print(spin_norm)
        spin = -axis * spin_norm[1]  # HACK: minus sign had to be added

        first_rot = self.quat2rot(qtls[0])
        return spin, valid_idx

    def get_RANSAC_spin_axis(self, y):
        n_rots = len(y)
        iter = 0
        best_idxs = None
        best_axis = None
        best_u = None
        best_err = 1e6
        # RANSAC Loop
        while iter < self.k:
            maybe_idx = np.sort(
                np.random.choice(np.arange(n_rots), self.n, replace=False)
            )
            maybe_y = y[maybe_idx]
            axis, u1, u2 = self.get_spin_axis(maybe_y)

            # print(maybe_idx)
            # print()
            also_inliers_idx = []
            for i in range(n_rots):
                if i not in maybe_idx:
                    q, res_error = self.get_qtls(y[i], u1, u2)
                    # print(res_error)
                    if res_error < self.t:
                        also_inliers_idx.append(i)
            also_inliers_idx = np.array(also_inliers_idx)
            # print(len(also_inliers_idx))
            maybe_idx = np.sort(np.concatenate((maybe_idx, also_inliers_idx)))
            # print(maybe_idx)
            if len(also_inliers_idx) > self.d:
                maybe_y = y[maybe_idx, :]
                axis, u1, u2 = self.get_spin_axis(maybe_y)
                tot_error = 0
                for i in maybe_idx:
                    q, res_error = self.get_qtls(y[i], u1, u2)
                    tot_error += res_error
                # print(tot_error)
                if tot_error < best_err:
                    best_axis = axis
                    best_idxs = maybe_idx
                    best_err = tot_error
                    best_u = (u1, u2)
            iter += 1
        return best_axis, best_u, best_idxs, best_err

    def rot2quat(self, rot):
        q = rot.as_quat()
        return np.array(
            [q[3], q[0], q[1], q[2]]
        )  # Different convention used for quaternion

    def quat2rot(self, q):
        rot = R.from_quat(
            [q[1], q[2], q[3], q[0]]
        )  # Different convention used for quaternion
        return rot


#### Functions for the tests #####


def predict(t, omega, orientation):
    orientations = [orientation]
    for i in range(1, len(t)):
        dt = t[i] - t[0]
        new_orientation = R.from_rotvec(dt * omega) * orientations[0]
        # print(quat_from_rot(new_orientation))
        orientations.append(new_orientation)
    return orientations


def residual_error(pred_rot, real_rot):
    # print("Real rot: {}".format(real_rot.as_rotvec()))
    # print("Pred_rot: {}".format(pred_rot.as_rotvec()))
    tot_error = 0
    if isinstance(pred_rot, list):
        n_rot = len(real_rot)
        for i in range(n_rot):
            error = (real_rot[i] * pred_rot[i].inv()).magnitude()
            tot_error += error
        tot_error = tot_error / n_rot
        return tot_error
    else:
        error = (real_rot * pred_rot.inv()).magnitude()
        return error


if __name__ == "__main__":
    # Test for the spin regression
    dt = 0.001
    print("dt = {}".format(dt))
    spin_regressor = SpinRegressor()
    # f is the spin frequency
    for f in range(10, 50, 5):
        # Parameters of the test
        gt_norm_omega = 2 * np.pi * f
        print("GT spin omega: {}".format(gt_norm_omega))
        gt_axis = np.array([1, 0, 0])
        gt_axis = gt_axis / np.linalg.norm(gt_axis)
        print("GT spin axis: {}".format(gt_axis))
        gt_phi = gt_norm_omega * dt
        print("GT phi: {}".format(gt_phi))
        gt_spin = np.array(gt_axis * gt_norm_omega)
        gt_spin_rot = R.from_rotvec(gt_spin * dt)
        print("GT spin: {}".format(gt_spin))
        print("GT spin rot: {}".format(gt_spin_rot.as_rotvec()))
        r0 = R.from_euler(seq="xyz", angles=[np.pi / 2, np.pi / 3, 0])
        r0 = R.from_euler(seq="xyz", angles=[0, 0, 0])
        print("Initial orientation: {} (euler angles)".format(r0.as_euler(seq="xyz")))
        print()

        # Generating the observations / rotations
        ts = dt * np.arange(0, 10)
        rs = predict(ts, gt_spin, r0)

        # Regress
        # spin, first_rot = spin_regressor.regress(ts, rs)

        t1 = time.time()
        spin, first_rot = spin_regressor.RANSAC_regress(ts, rs)
        print("Runtime: {}".format(time.time() - t1))

        if spin is None:
            raise RuntimeError("Test failed")

        # Check if prediction is correct
        pred_rot = predict(ts, spin, first_rot)
        error = residual_error(pred_rot, rs)

        # Print error
        print("Error: {}".format(error))
        print("Estimated spin")
        print(spin)
        print("Estimated first rot")
        print(first_rot.as_rotvec())
        print("Real first rot")
        print(rs[0].as_rotvec())

        print()
        print()
