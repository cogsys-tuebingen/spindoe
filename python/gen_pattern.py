"""
Script to generate an optimized dot pattern
It implements the hash table generation with pytorch in order to use gradient-based optimization
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import scipy
import scipy.spatial
from tqdm import tqdm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from utils import write_pattern


class HashTable(nn.Module):
    def __init__(self, n_points=20, **kwargs):
        super().__init__()
        # Randomly sample the initial dot pattern
        low = torch.tensor([0, 0], dtype=torch.float64)
        high = torch.tensor([np.pi, 2 * np.pi], dtype=torch.float64)
        sampler = torch.distributions.uniform.Uniform(low, high)
        self.sphere_points = sampler.sample(torch.Size([n_points]))
        self.sphere_points = torch.nn.parameter.Parameter(self.sphere_points)

        # Settings for loss function
        # The dots cannot be too near each other otherwise they might not be distinguishable
        self.d_min = 2e-1  # Smallest distance allowed between 2 nearest points
        # There needs to be at least 3 dots observable so they can't be too far from each other
        self.d_max = (
            np.pi / 2
        )  # Biggest distance allowed between a point and its 2 nearest neighbours
        self.relu = nn.ReLU()
        # To limit the number of entries in the hash table, we assume that dots
        # situated on opposite sides of the sphere cannot be observed at the same time
        self.max_dist_obs_pts = kwargs.get("min_feature_dist", 1 * np.sqrt(2))

    def sphere2cart(self, sphere_coord):
        """
        Converts spherical coordinates to cartesian coordinates
        (the points are assumed to be on the unit sphere)

        :param sphere_coord np.array: [theta, phi]
        """
        x = torch.cos(sphere_coord[:, 1]) * torch.sin(sphere_coord[:, 0])
        y = torch.sin(sphere_coord[:, 1]) * torch.sin(sphere_coord[:, 0])
        z = torch.cos(sphere_coord[:, 0])
        cart_coord = torch.vstack((x, y, z)).T
        return cart_coord

    def gen_hash_table(self, sph_points):
        """
        Generates the hash table for the pattern of sph_points

        :param sph_points tensor: Dot pattern described with spherical coordinates [theta, phi]
        """
        points = self.sphere2cart(sph_points)
        hash_table = torch.empty((0, 3))
        self.n_points = len(points)
        for i in range(self.n_points):
            for j in range(self.n_points):
                if i == j:
                    continue
                if torch.linalg.norm(points[i] - points[j]) > self.max_dist_obs_pts:
                    continue

                basis = self.gen_basis(points[[i, j]])
                for k in range(self.n_points):
                    if k == i or k == j:
                        continue
                    if torch.linalg.norm(points[i] - points[k]) > self.max_dist_obs_pts:
                        continue
                    feature = points[k]
                    value = self.cart2hash(basis, feature)
                    hash_table = torch.cat((hash_table, value[None, :]), dim=0)
        return points, hash_table

    def forward(self):
        points, hash_table = self.gen_hash_table(self.sphere_points)
        return points, hash_table

    def loss(self, points, hash_table):
        ## Loss for minmax in hash table
        self.hash_tree = scipy.spatial.KDTree(hash_table.clone().detach().numpy())
        d = 0
        for point in hash_table:
            np_point = point.clone().detach().numpy()
            _, nearest_pt_idx = self.hash_tree.query(np_point, 2)
            d += torch.sqrt(torch.sum((point - hash_table[nearest_pt_idx[1]]).pow(2)))
        d = d / hash_table.size(dim=0)

        ## Loss from proximity on sphere surface
        self.point_tree = scipy.spatial.KDTree(points.clone().detach().numpy())
        for point in points:
            np_point = point.clone().detach().numpy()
            _, nearest_pt_idx = self.point_tree.query(np_point, 4)
            for nearest in nearest_pt_idx[1:]:
                d -= self.relu(
                    self.d_min - torch.sqrt(torch.sum((point - points[nearest]).pow(2)))
                )
                d -= self.relu(
                    -self.d_max
                    + torch.sqrt(torch.sum((point - points[nearest]).pow(2)))
                )
        return -d

    def gen_basis(self, points):
        # The basis is the linear transform from the sphere frame to the hash frame
        basis_x = points[0]
        basis_y = points[1]
        basis_z = torch.cross(basis_x, basis_y)
        basis = torch.vstack((basis_x, basis_y, basis_z)).T
        return basis

    def cart2hash(self, basis, points):
        return torch.matmul(torch.inverse(basis), points.T).T

    def plot(self):
        points = self.sphere2cart(self.sphere_points)

        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")

        points = points.clone().detach().numpy()
        ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], color="red")
        self.plot_sphere(ax)
        # ax.set_aspect("equal")
        plt.title("Dot pattern")

        plt.show()

    def plot_sphere(self, ax):
        ## Draw a sphere surface that represents the ball
        r_ball = 1
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = r_ball * np.outer(np.cos(u), np.sin(v))
        y = r_ball * np.outer(np.sin(u), np.sin(v))
        z = r_ball * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(
            x, y, z, rstride=4, cstride=4, color="b", linewidth=0, alpha=0.2
        )


if __name__ == "__main__":
    torch.random.manual_seed(0)
    n_points = 20
    hash_table = HashTable(n_points=n_points)
    learning_rate = 1e-1
    optimizer = torch.optim.SGD(hash_table.parameters(), lr=learning_rate)
    for i in tqdm(range(100)):
        points, ht = hash_table.forward()
        loss = hash_table.loss(points, ht)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
    hash_table.plot()
    file_path = "points_opt.csv"
    write_pattern(file_path, points.clone().detach().numpy())
