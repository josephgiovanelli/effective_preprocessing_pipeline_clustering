
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

from hyperopt import hp

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
    "n_clusters": list(range(2, 201))
}

grid_mini_batch_k_means = {
    "n_clusters": list(range(2, 201))
}

grid_k_medoids = {
    "n_clusters": list(range(2, 201))
}

grid_gaussian_mixture = {
    "n_components": list(range(2, 201))
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


def get_domain_space(max_k=200):
    algorithms_space = []
    print('clustering')
    for algorithm in parameter_grid.keys():
        print(f'''\t{algorithm}''')
        algorithm_config = {}
        to_print_algorithm_config = ''
        for k, v in parameter_grid[algorithm].items():
            if k == "n_clusters" or k == "n_components" or k == "bandwidth":
                #v = list(range(2, max_k))
                v = list(range(2, 13))
            to_print_algorithm_config += '\t\t{}: {}\n'.format(k, v)
            algorithm_config[k] = hp.choice('{}_{}'.format(algorithm, k), v)
        algorithms_space.append((algorithm, algorithm_config))
        print(to_print_algorithm_config)
    return hp.choice('algorithm', algorithms_space)
