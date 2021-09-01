
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

from experiment.algorithm.utils import generate_domain_space, generate_union_domain_space

algorithms = {
    'KMeans': KMeans,
    'MiniBatchKMeans': MiniBatchKMeans,
    'KMedoids': KMedoids,
    'GaussianMixture': GaussianMixture,
    #'MeanShift': MeanShift,
    #'AgglomerativeClustering': AgglomerativeClustering,
    #'SpectralClustering': SpectralClustering,
    #'OPTICS': OPTICS,
    #'Birch': Birch,
}

grid_k_means = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

grid_mini_batch_k_means = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

grid_k_medoids = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

grid_gaussian_mixture = {
    "n_components": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

grid_mean_shift = {
    "bandwidth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

grid_agglomerative_clustering = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

parameter_grid = {
    'KMeans': grid_k_means,
    'MiniBatchKMeans': grid_mini_batch_k_means,
    'KMedoids': grid_k_medoids,
    'GaussianMixture': grid_gaussian_mixture,
    #'MeanShift': grid_mean_shift,
    #'AgglomerativeClustering': grid_agglomerative_clustering,
    #'SpectralClustering': grid_spectral_clustering,
    #'OPTICS': grid_optics,
    #'Birch': grid_birch,
}


def get_domain_space(algorithm_name):
    if algorithm_name in parameter_grid.keys():
        return generate_domain_space(parameter_grid.get(algorithm_name))
    else:
        return generate_union_domain_space(parameter_grid)
