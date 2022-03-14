from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import indices
import pandas as pd

from fsfc.generic import GenericSPEC

from sklearn.neighbors import LocalOutlierFactor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, PowerTransformer, KBinsDiscretizer, \
    Binarizer, OneHotEncoder, OrdinalEncoder, FunctionTransformer

from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.manifold import TSNE

from imblearn.pipeline import Pipeline
from imblearn.base import BaseSampler
from imblearn.utils import check_sampling_strategy

from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import check_classification_targets
from sklearn.datasets import make_blobs

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from s_dbw import S_Dbw

from experiment.pipeline.outlier_detectors import MyOutlierDetector
from experiment.utils import datasets
from experiment.utils.metrics import my_silhouette_samples

def plot(dataset, features, n_features, scaler, outlier, n_clusters, natural_clusters, internal_metric):
    
    X, y, original_features = datasets.get_dataset(dataset)

    '''
    X, y = make_blobs(n_samples=(400, 1000, 300, 300), n_features=2, centers=[(-20, -10), (-2.5, -10), (-7.5, -22), (0, -22.)], cluster_std=[3.5, 2.5, 1., 1.],  shuffle=True, random_state=42)
    X[:, 1] *= 10
    X[:, 0] *= 5
    np.random.seed(42)
    X = np.column_stack((X, np.random.uniform(low=5, high=100, size=(2000, 3))))
    

    original_features = [str(i) for i in range(X.shape[0])]
    data = np.column_stack((X, y))
    pd.DataFrame(data).to_csv("datasets/synthetic.csv", header=None, index=None)
    '''

    myFeatureEngineeringTransformer = GenericSPEC(k=n_features)
    myScaler = StandardScaler()
    myOutlierDetector = MyOutlierDetector(n_neighbors=32)
    myEstimator = KMeans(max_iter=10, n_clusters=n_clusters, random_state=42)
    pipe = Pipeline([ 
        ('features', myFeatureEngineeringTransformer if features else FunctionTransformer()),
        ('scaler', myScaler if scaler else FunctionTransformer()), 
        ('outlier', myOutlierDetector if outlier else FunctionTransformer()), 
        ('estimator', myEstimator)
        ])
    
    y_pred = pipe.fit_predict(X, y)
    if outlier:
        Xt, yt = pipe[:-1].fit_resample(X, y)
    else:
        Xt = pipe[:-1].fit_transform(X)
        yt = y
    
    if internal_metric == 'sil':
        internal_metric_value = silhouette_score(Xt, y_pred)
        #sil_samples, intra_clust_dists, inter_clust_dists = my_silhouette_samples(Xt, result)
    elif internal_metric == 'ch':
        internal_metric_value = calinski_harabasz_score(Xt, y_pred)
    elif internal_metric == 'dbi':
        internal_metric_value = -1 * davies_bouldin_score(Xt, y_pred)
    elif internal_metric == 'sdbw':
        internal_metric_value = -1 * S_Dbw(Xt, y_pred)
    elif internal_metric == 'ssw':
        internal_metric_value = -1 * pipe[-1].inertia_
    elif internal_metric == 'sw':
        _, intra_clust_dists, _ = my_silhouette_samples(Xt, y_pred)
        internal_metric_value = intra_clust_dists.sum()
        internal_metric_value = -1 * internal_metric_value
    print(f'internal metric: {internal_metric_value}')
    print(f'external metric: {metrics.adjusted_mutual_info_score(yt, y_pred)}')
    print(f'ssw: {myEstimator.inertia_}')

    #Xt = Xt[:, :2]
    if Xt.shape[1] > 2:
        Xt = TSNE(n_components=2, random_state=42).fit_transform(Xt)
        labels = ['TSNE_0', 'TSNE_1', 'TSNE_2']
    else:
        if features:
            indeces = myFeatureEngineeringTransformer.get_support()
            indeces = [i for i, e in enumerate(indeces) if e == True]
        else:
            indeces = range(len(original_features))
        labels = [original_features[i] for i in indeces]

    fig = plt.figure()
    colors = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'mediumspringgreen', 'peru'])
    markers = ['o', 'v', '1', 's', 'p', 'P', '*', 'X', 'D', '_']

    y_to_plot = yt if natural_clusters else y_pred
    min, max = Xt.min(), Xt.max()
    centers = myEstimator.cluster_centers_
    if Xt.shape[1] < 3:
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(Xt[:, 0], Xt[:, 1] if Xt.shape[1] > 1 else [(max+min)/2] * Xt.shape[0], c=[colors[int(i)] for i in y_to_plot])
        #ax.scatter(centers[:, 0], centers[:, 1] if Xt.shape[1] > 1 else [(max+min)/2] * Xt.shape[0], c='cyan', marker='*', s=10)
    else:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], c=[colors[int(i)] for i in y_to_plot])
        #ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='cyan', marker='*', s=10)


    #ax.set_xlim([min, max])
    #ax.set_ylim([min, max])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1] if Xt.shape[1] > 1 else 'None')
    if Xt.shape[1] == 3:
        #ax.set_zlim([min, max])
        ax.set_zlabel(labels[2])
    features_str = 'features_' if features else ''
    scaler_str = 'scaler_' if scaler else ''
    outlier_str = 'outlier_' if outlier else ''
    clustering_str = 'natural' if natural_clusters else 'pred'
    fig.tight_layout()
    #fig.savefig(dataset + '_' + str(n_features) +  'feat_tsne.pdf')
    plt.show()

#plot(dataset='synthetic', features=True, n_features=3, scaler=False, outlier=True, n_clusters=3, natural_clusters=True, internal_metric='sil')
#plot(dataset='synthetic', features=True, n_features=3, scaler=False, outlier=True, n_clusters=3, natural_clusters=False, internal_metric='sil')

#plot(dataset='synthetic', features=True, n_features=1, scaler=False, outlier=True, n_clusters=2, natural_clusters=True, internal_metric='sil')
#plot(dataset='synthetic', features=True, n_features=1, scaler=False, outlier=True, n_clusters=2, natural_clusters=False, internal_metric='sil')

#plot(dataset='synthetic', features=True, n_features=2, scaler=False, outlier=True, n_clusters=2, natural_clusters=True, internal_metric='sil')
#plot(dataset='synthetic', features=True, n_features=2, scaler=False, outlier=True, n_clusters=2, natural_clusters=False, internal_metric='sil')

#plot(dataset='synthetic', features=True, n_features=2, scaler=True, outlier=True, n_clusters=4, natural_clusters=True, internal_metric='sil')
#plot(dataset='synthetic', features=True, n_features=2, scaler=True, outlier=True, n_clusters=4, natural_clusters=False, internal_metric='sil')

#plot(dataset='synthetic', features=True, n_features=2, scaler=True, outlier=False, n_clusters=4, natural_clusters=True, internal_metric='sil')
#plot(dataset='synthetic', features=True, n_features=2, scaler=True, outlier=False, n_clusters=4, natural_clusters=False, internal_metric='sil')

#plot(dataset='synthetic', features=True, n_features=2, scaler=False, outlier=False, n_clusters=4, natural_clusters=True, internal_metric='sil')
#plot(dataset='synthetic', features=True, n_features=2, scaler=False, outlier=False, n_clusters=4, natural_clusters=False, internal_metric='sil')

#plot(dataset='synthetic', features=False, n_features=5, scaler=False, outlier=False, n_clusters=4, natural_clusters=True, internal_metric='sil')
#plot(dataset='synthetic', features=False, n_features=5, scaler=False, outlier=False, n_clusters=4, natural_clusters=False, internal_metric='sil')

#plot(dataset='seeds', features=True, n_features=1, scaler=False, outlier=False, n_clusters=3, natural_clusters=True, internal_metric='sil')
#plot(dataset='seeds', features=True, n_features=2, scaler=False, outlier=False, n_clusters=3, natural_clusters=True, internal_metric='sil')
#plot(dataset='seeds', features=True, n_features=3, scaler=False, outlier=False, n_clusters=3, natural_clusters=True, internal_metric='sil')
#plot(dataset='seeds', features=True, n_features=4, scaler=False, outlier=False, n_clusters=3, natural_clusters=True, internal_metric='sil')
#plot(dataset='seeds', features=True, n_features=5, scaler=False, outlier=False, n_clusters=3, natural_clusters=True, internal_metric='sil')
#plot(dataset='seeds', features=True, n_features=6, scaler=False, outlier=False, n_clusters=3, natural_clusters=True, internal_metric='sil')
plot(dataset='synthetic', features=False, n_features=1, scaler=False, outlier=False, n_clusters=2, natural_clusters=False, internal_metric='sil')

'''
plot silhouette chart
if internal_metric == 'SIL':
            fig, ax = plt.subplots(1, 3)
            fig.set_size_inches(18, 7)
            n_clusters = y_pred.iloc[:, 0].unique().size
            my_silhouette_values = pd.DataFrame({
                'silhouette': sil_samples, 
                'inter_dists': inter_clust_dists, 
                'intra_dists': intra_clust_dists, 
                'y_pred': result,})
            for j in range(len(ax)):
                y_lower = 10
                ax[j].set_xlim([0, my_silhouette_values[['inter_dists', 'intra_dists']].max().max()])
                ax[j].set_yticks([])
                ax[j].set_ylim([0, Xt.shape[0] + (n_clusters + 1) * 10])
                ax[j].set_title(my_silhouette_values.columns[j])
                ax[j].set_xlabel("values")
                for i in range(n_clusters):
                    ith_cluster_silhouette_values = my_silhouette_values[my_silhouette_values['y_pred'] == i]
                    ith_cluster_silhouette_values = ith_cluster_silhouette_values.sort_values(by=['silhouette'])
                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                        
                    ax[j].fill_betweenx(
                        np.arange(y_lower, y_upper),
                        0,
                        ith_cluster_silhouette_values.iloc[:, j],
                        facecolor=colors[i],
                        edgecolor=colors[i],
                        alpha=0.7,
                    )

                    if j == 0:
                        ax[j].set_xlim([my_silhouette_values['silhouette'].min() - 0.1, 1])
                        ax[j].axvline(x=score, color="red", linestyle="--")  
                        ax[j].text(my_silhouette_values['silhouette'].min() - 0.05, y_lower + 0.5 * size_cluster_i, str(i)) 
                        ax[j].set_ylabel("Cluster labels")
                    y_lower = y_upper + 10  
            plt.tight_layout()
            fig.savefig(os.path.join(details_path, file_name + "_silhouette.png"))
'''

    