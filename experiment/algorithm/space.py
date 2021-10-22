
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

from experiment.algorithm.utils import generate_domain_space, generate_union_domain_space

algorithms = {
    'KMeans': KMeans,
    #'MiniBatchKMeans': MiniBatchKMeans,
    #'KMedoids': KMedoids,
    #'GaussianMixture': GaussianMixture,

    #'MeanShift': MeanShift,
    #'AgglomerativeClustering': AgglomerativeClustering,
    #'SpectralClustering': SpectralClustering,
    #'OPTICS': OPTICS,
    #'Birch': Birch,
}

grid_k_means = {
    "n_clusters": list(range(2, 201)),
    "max_iter": [10]
}

grid_mini_batch_k_means = {
    "n_clusters": list(range(2, 201)),
    "max_iter": [10]
}

grid_k_medoids = {
    "n_clusters": list(range(2, 201)),
    "max_iter": [10]
}

grid_gaussian_mixture = {
    "n_components": list(range(2, 201)),
    "max_iter": [10]
}

grid_mean_shift = {
    "bandwidth": list(range(2, 201))
}

grid_agglomerative_clustering = {
    "n_clusters": list(range(2, 201))
}

parameter_grid = {
    'KMeans': grid_k_means,
    #'MiniBatchKMeans': grid_mini_batch_k_means,
    #'KMedoids': grid_k_medoids,
    #'GaussianMixture': grid_gaussian_mixture,
        
    #'MeanShift': grid_mean_shift,
    #'AgglomerativeClustering': grid_agglomerative_clustering,
    #'SpectralClustering': grid_spectral_clustering,
    #'OPTICS': grid_optics,
    #'Birch': grid_birch,
}


def get_domain_space(algorithm_name, max_k=200):
    if algorithm_name in parameter_grid.keys():
        return generate_domain_space(parameter_grid.get(algorithm_name, max_k))
    else:
        return generate_union_domain_space(parameter_grid, max_k)
