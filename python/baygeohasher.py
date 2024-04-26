"""
Bayesian Geometric Hasher
Based on:  https://www.researchgate.net/publication/3344381_Geometric_Hashing_An_Overview

"""

import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
from kent_distr import kent
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils import read_pattern, write_pattern


class BayGeoHasher:
    def __init__(
        self,
        threshold: float = -5e4,
        k_neighbour: int = 20,
        upper_score_perc: int = 50,
        projection_ll_scale: float = 0.03,
        min_feature_dist: float = 3,  # HACK: should be 1.8
        kappa: float = 5e2,
        beta: float = 0,
    ) -> None:
        self.threshold = threshold
        self.k_neighbour = k_neighbour
        self.upper_score_perc = upper_score_perc
        self.projection_ll_scale = projection_ll_scale
        self.min_feature_dist = min_feature_dist

        # Parameter for the kent distribution reprensenting the dots' position
        self.kappa = kappa
        self.beta = beta
        if self.kappa < 0:
            raise ValueError("kappa has to be strictly greater than 0")
        if 0 > 2 * self.beta or 2 * self.beta > self.kappa:
            raise ValueError("2*Beta has to be lower than kappa and greater than 0")

    def gen_hash_table(self, ref_points):
        """
        Generates the hash table necessary to later identify the dots

        :param ref_points np.array: Reference pattern of size [n x 3]
        """
        self.n_points = ref_points.shape[0]
        self.ref_points = ref_points
        self.hash_table = []
        self.bases = []
        for i in range(self.n_points):
            for j in range(self.n_points):
                if i == j:
                    continue
                # If the 2 basis points are not likely to be observed at the same time
                # (opposite sides of the ball), this basis is ignored
                if (
                    np.linalg.norm(ref_points[i] - ref_points[j])
                    > self.min_feature_dist
                ):
                    continue

                basis = self.gen_basis(ref_points[[i, j]])
                for k in range(self.n_points):
                    if k == i or k == j:
                        continue
                    if (
                        np.linalg.norm(ref_points[i] - ref_points[k])
                        > self.min_feature_dist
                    ):
                        continue

                    feature = ref_points[k]
                    value = self.cart2hash(basis, feature)
                    self.hash_table.append(value)
                    self.bases.append([i, j])

        self.hash_table = np.array(self.hash_table)
        self.bases = np.array(self.bases)
        # We create KDTrees to make easier to look up the closest points
        self.hash_tree = scipy.spatial.KDTree(self.hash_table)
        self.point_tree = scipy.spatial.KDTree(self.ref_points)

    def gen_basis(self, basis_points):
        basis_x = basis_points[0]
        basis_y = basis_points[1]
        basis_z = np.cross(basis_x, basis_y)
        basis = np.array([basis_x, basis_y, basis_z]).T
        return basis

    def asSpherical(self, xyz):
        """
        Returns the spherical coordinates of a vector
        It is a bit different from the standard implementation, theta is the angle
        between X and the ZY plane instead of the angle between Z and the XY plane
        :param xyz np.array: cartesian coordinate vector
        """
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(x / r)
        phi = np.arctan2(z, y)
        return (r, theta, phi)

    def get_log_ll(self, basis, ball_point, hash_values):
        """
        Returns the log likelihood that the hash values were generated by ball_point using basis

        :param basis np.array: Basis matrix used transform ball point into the hash space``
        :param ball_point np.array: Cartesian coordinate vector of ball_point
        :param hash_values np.array: Array of hash values [n x 3]
        """
        # Get likelihood of a feature being certain hash_values
        p = []
        _, theta, phi = self.asSpherical(ball_point)
        kent_distr = kent(theta, phi, 0, self.kappa, self.beta)
        cart_hashs = self.hash2cart(basis, hash_values)
        norms = np.linalg.norm(cart_hashs, axis=1)
        cart_hashs = np.divide(cart_hashs.T, norms).T
        sphere_lls = kent_distr.pdf(cart_hashs) * np.abs(
            np.linalg.det(basis)
        )  # normalizing factor
        proj_lls = scipy.stats.norm.pdf(norms, loc=1, scale=self.projection_ll_scale)
        lls = np.multiply(sphere_lls, proj_lls)
        ps = np.maximum(lls, 1e-10)
        return np.log(ps)

    def cart2hash(self, basis, points):
        """
        Converts the points into the hash space using basis for the transformation

        :param basis np.array: Basis matrix [3 x 3]
        :param points np.array: Array of hash values [n x 3]
        """
        hash_points = (np.linalg.inv(basis) @ points.T).T
        return hash_points

    def hash2cart(self, basis, hash_values):
        """
        Converts a hash value back into the cartesian space using basis for the transformation

        :param basis np.array: Basis matrix [3 x 3]
        :param hash_values np.array: Array of hash values [n x 3]
        """
        cart_points = (basis @ hash_values.T).T
        return cart_points

    def identify(self, obs_points):
        """
        Returns the idx of the first 2 observed dots, the corresponding
        rotation from the reference pattern to the observed pattern and reprojection error

        :param obs_points np.array: Array of points [n x 3]
        """
        if self.hash_table is None:
            raise Exception("Hash table wasn't generated")
        if obs_points.shape[0] < 3:
            raise Exception(
                "3 points are required minimum, only {} were given".format(
                    obs_points.shape[0]
                )
            )
        score = np.full((self.n_points, self.n_points), np.nan)
        basis = self.gen_basis(obs_points[:2])
        features = self.cart2hash(basis, obs_points[2:])

        for i in range(len(features)):
            d, neighbour_idx = self.hash_tree.query(features[i], self.k_neighbour)
            # print(self.bases[neighbour_idx])
            p = self.get_log_ll(
                basis, obs_points[i + 2], self.hash_table[neighbour_idx]
            )
            valid = p > self.threshold
            # print(valid)
            for j in range(self.k_neighbour):
                if valid[j]:
                    base_i = self.bases[neighbour_idx[j]][0]
                    base_j = self.bases[neighbour_idx[j]][1]
                    # print("{}    {}".format(base_i, base_j))
                    if np.isnan(score[base_i, base_j]):
                        score[base_i, base_j] = 0
                    score[base_i, base_j] += p[j]
        score_lim = np.nanpercentile(score, self.upper_score_perc)
        valid_idx = np.argwhere(score >= score_lim)
        valid_score = score[score >= score_lim]
        # print(valid_idx)
        # print(valid_score)
        best_idx = valid_idx[np.argmax(valid_score)]
        # print(best_idx)
        # print(len(obs_points))
        # print(score_lim)
        # print(check_models)
        # print()

        ## Verification phase
        errors = []
        rots = []
        for idx in valid_idx:
            rot, rmse = self.verification(idx[0], idx[1], obs_points)
            errors.append(rmse)
            rots.append(rot)
            # print(errors[-1])
            # print(sel_basis)
        chosen_base = valid_idx[np.argmin(errors)]
        chosen_rot = rots[np.argmin(errors)]
        chosen_error = np.min(errors)
        return chosen_base, chosen_rot, chosen_error

        ## Without verification phase
        # rot, rmse = self.verification(best_idx[0], best_idx[1], obs_points)
        # return best_idx, rot, rmse

    def verification(self, i, j, obs_points):
        """
        Check the rmse of the observed points compared with the reference points when
        using as basis ref_points i and j
        It returns the estimate rotation using Kalbsch's algorithm and the corresponding rmse
        It is the rotation that transform the reference pattern into the observed pattern

         :param i int: idx of the 1st point used for the generating the basis
         :param j int: idx of the 2nd point used for generating the basis
         :param obs_points np.array: Array of the observed points [n x 3]
        """

        ref_points = np.array([self.ref_points[i], self.ref_points[j]])
        # Get a first approximation of the rotation with the basis
        rot, _ = R.align_vectors(ref_points, obs_points[:2])
        corresponding_ref_points = []
        for point in obs_points[2:]:
            rot_point = rot.apply(point)
            d, nearest_point_idx = self.point_tree.query(rot_point, 1)
            # print(d)
            corresponding_ref_points.append(self.ref_points[nearest_point_idx])
        corresponding_ref_points = np.array(corresponding_ref_points)
        ref_points = np.concatenate([ref_points, corresponding_ref_points], axis=0)

        # Get a better approximation
        rot, rmse = R.align_vectors(ref_points, obs_points)

        corresponding_ref_points_idx = [i, j]
        for point in obs_points[2:]:
            rot_point = rot.apply(point)
            d, nearest_point_idx = self.point_tree.query(rot_point, 1)
            if point[2] > 0.3:
                corresponding_ref_points_idx.append(nearest_point_idx)

        corresponding_ref_points_idx = np.sort(np.array(corresponding_ref_points_idx))
        rotated_ref_points = rot.inv().apply(self.ref_points)
        obs_dot_idx = np.sort(np.argwhere(rotated_ref_points[:, 2] > 0.3).flatten())
        # print(obs_dot_idx)
        # print(corresponding_ref_points_idx)
        obs_dot_idx = set(obs_dot_idx)
        corresponding_ref_points_idx = set(corresponding_ref_points_idx)
        n_diff = len(obs_dot_idx - corresponding_ref_points_idx)
        # print(obs_dot_idx - corresponding_ref_points_idx)
        # print(n_diff)
        # print(rmse)
        # if n_diff > 0:
        #     rmse = rmse * (1 + n_diff / len(obs_dot_idx))
        # print(rmse)
        # error = utils.get_P2P_distance(ref_points, rot.apply(obs_points), 1)
        # print("Error comparison")
        # print(np.linalg.norm(error))
        # print(rmse)
        return rot, rmse


def observable_dots(point_list, d_thres):
    """
    Returns the indices of the points that are observable assuming the ball is watched from the top

    :param point_list np.array: pattern points
    :param d_thress float: Distance threshold from where we assume the dot is not observable
    """
    top_sphere = np.array([0, 0, 1])
    d = np.linalg.norm(point_list - top_sphere, axis=1)
    sorted_indices = np.argsort(d)
    indices = np.array([i for i in sorted_indices if d[i] < d_thres])
    return indices


if __name__ == "__main__":
    # Test that the bayesian geometric hashing works with perfect data
    # There might be some cases where there are not enough points observed
    n_tests = 1000

    np.random.seed(0)
    ref_pattern = read_pattern("../data/ref_pattern.csv")

    geohasher = BayGeoHasher()
    geohasher.gen_hash_table(ref_pattern)

    successes = 0
    few_features_problem = 0
    avg_features = 0
    t0 = time.time()
    for _ in tqdm(range(n_tests)):
        random_rotation = R.random()
        all_dots_rotated = random_rotation.apply(ref_pattern)
        center_point = np.array([0, 0, 1])
        idx = observable_dots(all_dots_rotated, 0.8 * np.sqrt(2))
        # idx = np.random.choice(np.arange(12), 3, replace=False)
        avg_features += len(idx) / n_tests
        if len(idx) < 3:
            # print("Number of points: ", len(idx))
            few_features_problem += 1
            continue
        features = all_dots_rotated[idx]
        basis_found, rot, error = geohasher.identify(features)
        success = (tuple(idx[:2]) == basis_found).all()
        # print()
        # print("{} | {}".format(basis_found, idx[:2]))
        # if success:
        #     print(True)
        # else:
        #     print(False)
        # print()
        successes += int(success)
    t_tot = time.time() - t0
    print("Time for a single run: {}".format(t_tot / n_tests))
    success_rate = successes / n_tests * 100
    print("Test: recognise random view")
    print("Success rate: ", success_rate, "%")
    print("Average features: ", avg_features)
    print("Too few features: ", few_features_problem / n_tests * 100, "%")
    print("")
