
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn_extra.cluster import KMedoids

from experiment.algorithm.utils import generate_domain_space, generate_union_domain_space

algorithms = {
    #'KMeans': KMeans,
    #'MiniBatchKMeans': MiniBatchKMeans,
    #'KMedoids': KMedoids,
    #'GaussianMixture': GaussianMixture,
    'RandomForest': RandomForestClassifier,
    #'MeanShift': MeanShift,
    #'AgglomerativeClustering': AgglomerativeClustering,
    #'SpectralClustering': SpectralClustering,
    #'OPTICS': OPTICS,
    #'Birch': Birch,
}

# 4800
grid_random_forest = {
    "n_estimators": [10, 25, 50, 75, 100],#, 150, 200],
    "max_depth": [1, 2, 3, 4, None],
    "max_features": [1, 2, 3, None],
    "min_samples_split": [2, 3, 5],
    #"min_weight_fraction_leaf": [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    'max_leaf_nodes': [2, 3, 5, None],
    #'min_impurity_decrease': [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    "bootstrap": [True, False],
    #"oob_score": [True, False]
    "criterion": ["gini", "entropy"],
    #"class_weight": [None, "balanced"]
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
    #'KMeans': grid_k_means,
    #'MiniBatchKMeans': grid_mini_batch_k_means,
    #'KMedoids': grid_k_medoids,
    #'GaussianMixture': grid_gaussian_mixture,
    'RandomForest': grid_random_forest,
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
