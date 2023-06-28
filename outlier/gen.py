import numpy as np
import matplotlib.pyplot as plt


def generate_noise_points(num_points, x_range, y_range):
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.random.uniform(y_range[0], y_range[1], num_points)
    return np.column_stack((x, y))

def gen_gaussian(n_cluster, center, cov, n_sample, n_noise):
    if n_cluster == 1:
        sample = np.random.multivariate_normal(center[0], cov[0], n_sample)
    
    if n_cluster == 2:
        cluster1_sample = np.random.multivariate_normal(center[0], cov[0], n_sample)
        cluster2_sample = np.random.multivariate_normal(center[1], cov[1], n_sample)
        sample = np.vstack((cluster1_sample, cluster2_sample))

    if n_cluster == 3:
        cluster1_sample = np.random.multivariate_normal(center[0], cov[0], n_sample)
        cluster2_sample = np.random.multivariate_normal(center[1], cov[1], n_sample)
        cluster3_sample = np.random.multivariate_normal(center[2], cov[2], n_sample)

        sample = np.vstack((np.vstack((cluster1_sample, cluster2_sample)), cluster3_sample))

    if n_cluster == 4:
        cluster1_sample = np.random.multivariate_normal(center[0], cov[0], n_sample)
        cluster2_sample = np.random.multivariate_normal(center[1], cov[1], n_sample)
        cluster3_sample = np.random.multivariate_normal(center[2], cov[2], n_sample)
        cluster4_sample = np.random.multivariate_normal(center[3], cov[3], n_sample)

        sample = np.vstack((np.vstack((np.vstack((cluster1_sample, cluster2_sample)), cluster3_sample)), cluster4_sample))

    x_range = [-7, 7]
    y_range = [-7, 7]
    noise_sample = generate_noise_points(n_noise, x_range, y_range)
    return sample, noise_sample

# n_cluster = 1
center = [[0, 0]]
cov = [[[0.5, 0.2], [0.2, 0.5]]]
sample, noise_sample = gen_gaussian(1, center, cov, 200, 10)

# n_cluster = 2
# center = [[1.5, 1.5], [-1.5, -1.5]]
# cov = [[[0.5, 0.2], [0.2, 0.5]], [[0.3, 0], [0, 0.3]]]
# sample, noise_sample = gen_gaussian(2, center, cov, 200, 10)

# n_cluster = 3
# center = [[2, 0], [-1.5, -2], [-1.5, 2]]
# cov = [[[0.5, 0.2], [0.2, 0.5]], [[0.3, 0], [0, 0.3]], [[0.4, 0.1], [0.2, 0.4]]]
# sample, noise_sample = gen_gaussian(3, center, cov, 200, 10)

# n_cluster = 4
# center = [[2, 2], [-2, -2], [-2, 2], [2, -2]]
# cov = [[[0.5, 0.2], [0.2, 0.3]], [[0.2, 0], [0, 0.4]], [[0.4, 0.1], [0.1, 0.4]], [[0.25, 0.2], [0.2, 0.25]]]
# sample, noise_sample = gen_gaussian(4, center, cov, 200, 10)

dataset = np.vstack((sample, noise_sample))
np.savetxt("./data/cluster_1", dataset, fmt="%.4f")
x = sample[:, 0]
y = sample[:, 1]
plt.scatter(x, y, c='k', marker='.', label='Gaussian Samples')

x = noise_sample[:, 0]
y = noise_sample[:, 1]
plt.scatter(x, y, c='k', marker='.', label='Noise Samples')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Clusters with Random Outlier')
# plt.grid(True)
plt.legend()
plt.show()
plt.savefig('cluster_1.png')