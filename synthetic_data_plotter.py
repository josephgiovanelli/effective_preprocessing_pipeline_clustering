from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from experiment.pipeline.outlier_detectors import MyOutlierDetector

from fsfc.generic import GenericSPEC

from sklearn.neighbors import LocalOutlierFactor

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn import metrics

from imblearn.pipeline import Pipeline
from imblearn.base import BaseSampler

from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import check_classification_targets
from imblearn.utils import check_sampling_strategy

from experiment.pipeline.outlier_detectors import MyOutlierDetector
from sklearn.preprocessing._function_transformer import FunctionTransformer

def objective(features, scaler, outlier):
    data = pd.read_csv('datasets/synthetic_data_outlier.csv', header=None)
    X, y = data.iloc[:, :-1],  data.iloc[:, -1]

    myFeatureEngineeringTransformer = GenericSPEC(k=3)
    myOutlierDetector = MyOutlierDetector(n_neighbors=32)
    myScaler = StandardScaler()
    myEstimator = KMeans(n_clusters=6, init='k-means++', random_state=0, algorithm='full')
    #myEstimator = KMedoids(n_clusters=n_clusters, method='pam', init='k-medoids++')
    #myEstimator = DBSCAN()

    pipe = Pipeline([ 
        ('features', myFeatureEngineeringTransformer if features else  FunctionTransformer()),
        ('scaler', myScaler if scaler else  FunctionTransformer()), 
        ('outlier', myOutlierDetector if outlier else  FunctionTransformer()), 
        ('estimator', myEstimator)
        ])

    y_pred = pipe.fit_predict(X.copy(), y.copy())
    if not(features) and not(scaler) and not(outlier):
        X, y = pd.DataFrame(X), pd.DataFrame(y)
    else:
        if outlier:
            X, y = pipe[:-1].fit_resample(X, y)
        else:
            X = pipe[:-1].fit_transform(X)
        X, y = pd.DataFrame(X), pd.DataFrame(y)
    #print(X, y)
    #print(pd.DataFrame(y_pred))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan'])
    markers = ['o', 'v', '1', 's', 'p', 'P', '*', 'X', 'D', '_']
    #ax.scatter(X.iloc[:, 0], X.iloc[:, 1],  X.iloc[:, 2], c=[colors[int(i)] for i in y.to_numpy()])
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1],  X.iloc[:, 2], c=[colors[int(i)] for i in y_pred])

    centers = pd.DataFrame(myEstimator.cluster_centers_)
    #ax.set_xlim([-20, 120])
    #ax.set_ylim([-20, 120])
    #ax.set_zlim([-20, 120])
    ax.scatter(centers.iloc[:, 0], centers.iloc[:, 1], centers.iloc[:, 2], c='cyan', marker='*', s=100)

    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    file_name = 'fixed_pipeline_' + '_'.join([str(features), str(scaler), str(outlier)])
    fig.savefig('plots/' + file_name + '.png')
    plt.close('all')
    #plt.show()
    
    return metrics.accuracy_score(y, y_pred), metrics.adjusted_mutual_info_score(y.iloc[:, 0].to_numpy(), y_pred), metrics.silhouette_score(X, y_pred), metrics.calinski_harabasz_score(X, y_pred), -1 * metrics.davies_bouldin_score(X, y_pred), file_name

results = pd.DataFrame()
i = 0
for features in [True, False]:
    for scaler in [True, False]:
        for outlier in [True, False]:
            accuracy, ami, silhouette, calinski, davies_bouldin, file_name = objective(features, scaler, outlier)
            results = results.append({
                "features": features, 
                "scaler": scaler, 
                "outlier": outlier,
                "file_name": file_name, 
                "accuracy": accuracy,
                'ami': ami,
                'silhouette': silhouette,
                'calinski': calinski,
                'davies_bouldin': davies_bouldin
            }, ignore_index=True)
            i += 1
            print(i)
            results.to_csv('plots/fixed_pipeline_results.csv', index=False)

results = results.sort_values(by=['silhouette', 'ami', 'accuracy'], ascending=False)
results.to_csv('fixed_pipeline_results.csv', index=False)


    