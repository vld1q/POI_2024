import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

__round__ = 2



def ransac(points, iterations=100):
    """

    :param points: The input array of points, where each point is represented as a 1-dimensional array-like structure.
    :param iterations: The number of iterations to run the RANSAC algorithm. Default is 100.
    :return: The best-fit plane, represented as a tuple (normal, point), where normal is the normal vector of the plane and point is a point on the plane.

    """
    eps = 0.01
    best_plane = None
    best_inliers = []
    best_error = np.inf

    for _ in range(iterations):
        random_indices = np.random.choice(points.shape[0], 3, replace=False)
        pts = points[random_indices]
        zero, vs = pts[0], pts[1:]
        vs = [v - zero for v in vs]

        if any(v.ndim != 1 for v in vs):
            continue
        normal = np.cross(*vs)
        normal /= np.linalg.norm(normal)

        distances = np.abs(np.dot(points - zero, normal))
        inliers = points[distances < eps]

        error = np.mean(distances[distances < eps])

        if len(inliers) > len(best_inliers) and error < best_error:
            best_plane = normal, zero
            best_inliers = inliers
            best_error = error

    return best_plane


load = lambda filenames: [np.array([[float(v) for v in line.strip().split()] for line in open(filename, 'r')]) for
                          filename in filenames]


def plot(clouds, planes, clusters):
    """
    Plot the given point clouds, planes, and clusters in a 3D plot.

    :param clouds: List of arrays representing the point clouds. Each array must have shape (N, 3), where N is the number of points in the cloud.
    :param planes: List of tuples representing the planes. Each tuple must have two elements: the normal vector of the plane and a point on the plane. The normal vector must have shape (
    *3,) and the point must have shape (3,).
    :param clusters: List of arrays representing the cluster labels for each point cloud. Each array must have shape (N,), where N is the number of points in the cloud.
    :return: None

    Example Usage:
    clouds = [cloud1, cloud2, cloud3]
    planes = [(plane1_normal, plane1_point), (plane2_normal, plane2_point), (plane3_normal, plane3_point)]
    clusters = [cluster1_labels, cluster2_labels, cluster3_labels]
    plot(clouds, planes, clusters)
    """
    fig = plt.figure(figsize=(18, 6))

    for i, (cloud, plane, cluster_labels) in enumerate(zip(clouds, planes, clusters), start=1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        ax.scatter(*[cloud[:, i] for i in range(3)], c=cluster_labels)
        normal, point = plane
        a, b, c = normal
        d = -point.dot(normal)
        xx, yy = np.meshgrid(*[np.linspace(min(cloud[:, i]), max(cloud[:, i]), 10) for i in range(2)])
        z = (-a * xx - b * yy - d) * 1. / c

        ax.plot_surface(xx, yy, z, alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point-Cloud no {i}')

    plt.show()

if __name__ == '__main__':
    clouds_xyz = load([f for f in os.listdir('.') if f.endswith('.xyz')])
    planes = [ransac(cloud) for cloud in clouds_xyz]
    kmeans_results = [KMeans(n_clusters=3).fit_predict(cloud) for cloud in clouds_xyz]
    plot(clouds_xyz, planes, kmeans_results)


    for i, cloud in enumerate(clouds_xyz):
        plane = ransac(cloud)
        normal = plane[0]
        print(
            f"Point-Cloud no {i + 1}:"
            f"\n\thas normal vector {[round(n, __round__) for n in normal]}"
            f"\n\tfound plane is {'horizontal' if all(np.abs(normal[a] > normal[b]) for a, b in [(2, 0), (2, 1)]) else 'vertical'}"
            f"\n\taverage points distance from the plane is {round(np.mean(np.abs(np.dot(cloud - plane[1], normal))), __round__)}"
        )
