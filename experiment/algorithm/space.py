
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture

from experiment.algorithm.utils import generate_domain_space, generate_union_domain_space

algorithms = {
    'KMeans': KMeans,
    'MiniBatchKMeans': MiniBatchKMeans,
    'MeanShift': MeanShift,
    #'SpectralClustering': SpectralClustering,
    'AgglomerativeClustering': AgglomerativeClustering,
    #'OPTICS': OPTICS,
    #'Birch': Birch,
    'GaussianMixture': GaussianMixture
}

grid_k_means = {
    "n_clusters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

grid_mini_batch_k_means = {
    "n_clusters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

grid_mean_shift = {
    "bandwidth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

grid_agglomerative_clustering = {
    "n_clusters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

grid_gaussian_mixture = {
    "n_components": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

parameter_grid = {
    'KMeans': grid_k_means,
    'MiniBatchKMeans': grid_mini_batch_k_means,
    'MeanShift': grid_mean_shift,
    #'SpectralClustering': grid_spectral_clustering,
    'AgglomerativeClustering': grid_agglomerative_clustering,
    #'OPTICS': grid_optics,
    #'Birch': grid_birch,
    'GaussianMixture': grid_gaussian_mixture
}


def get_domain_space(algorithm_name):
    if algorithm_name in parameter_grid.keys():
        return generate_domain_space(parameter_grid.get(algorithm_name))
    else:
        return generate_union_domain_space(parameter_grid)
