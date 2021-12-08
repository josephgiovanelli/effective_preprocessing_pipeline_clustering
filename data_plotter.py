from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fsfc.generic import GenericSPEC

from sklearn.neighbors import LocalOutlierFactor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, PowerTransformer, KBinsDiscretizer, \
    Binarizer, OneHotEncoder, OrdinalEncoder, FunctionTransformer

from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn import metrics

from imblearn.pipeline import Pipeline
from imblearn.base import BaseSampler

from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import check_classification_targets
from imblearn.utils import check_sampling_strategy

from experiment.pipeline.outlier_detectors import MyOutlierDetector
from sklearn.preprocessing._function_transformer import FunctionTransformer

def objective(features, scaler, outlier):
    data = pd.read_csv('datasets/synthetic_data.csv', header=None)
    X, y = data.iloc[:, :-1],  data.iloc[:, -1]
    X.to_csv('X.csv', index=False)
    y.to_csv('y.csv', index=False)

    numerical_features = [i for i in range(6)]
    categorical_features = []
    if features:
        myFeatureEngineeringTransformer = GenericSPEC(k=1)
        myFeatureEngineeringTransformer.fit(X)
        indeces = myFeatureEngineeringTransformer.get_support()
        print(indeces)
        new_i = 0
        new_numerical_features, new_categorical_features = [], []
        for i in range(len(indeces)):
            if indeces[i] == True:
                if i in numerical_features:
                    new_numerical_features += [new_i]
                else: #if i in self.current_categorical_features:
                    new_categorical_features += [new_i]
                new_i += 1
        numerical_features = new_numerical_features
        categorical_features = new_categorical_features

    myScaler = ColumnTransformer(
                        transformers=[
                            ('num', Pipeline(steps=[('normalizing', StandardScaler())]),
                             numerical_features),
                            ('cat', Pipeline(steps=[('identity_categorical', FunctionTransformer())]),
                             categorical_features)])

    myOutlierDetector = MyOutlierDetector(n_neighbors=32)

    myEstimator = KMeans(max_iter=10, n_clusters=3)
    #myEstimator = KMeans(n_clusters=6, init='k-means++', random_state=0, algorithm='full')
    #myEstimator = KMedoids(n_clusters=n_clusters, method='pam', init='k-medoids++')
    #myEstimator = DBSCAN()

    pipe = Pipeline([ 
        ('features', myFeatureEngineeringTransformer if features else  FunctionTransformer()),
        ('scaler', myScaler if scaler else  FunctionTransformer()), 
        ('outlier', myOutlierDetector if outlier else  FunctionTransformer()), 
        ('estimator', myEstimator)
        ])
    
    y_pred = pipe.fit_predict(X.copy(), y.copy())
    pd.DataFrame(y_pred).to_csv('y_pred.csv', index=False)
    if not(features) and not(scaler) and not(outlier):
        X, y = pd.DataFrame(X), pd.DataFrame(y)
    else:
        if outlier:
            X, y = pipe[:-1].fit_resample(X, y)
        else:
            X = pipe[:-1].fit_transform(X)
        X, y = pd.DataFrame(X), pd.DataFrame(y)
    print(X, y)
    #print(pd.DataFrame(y_pred))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan'])
    markers = ['o', 'v', '1', 's', 'p', 'P', '*', 'X', 'D', '_']
    #ax.scatter(X.iloc[:, 0], X.iloc[:, 1],  X.iloc[:, 2], c=[colors[int(i)] for i in y.to_numpy()])
    ax.scatter(X.iloc[:, 0], 0,  0, c=[colors[int(i)] for i in y_pred])

    #centers = pd.DataFrame(myEstimator.cluster_centers_)
    min, max = X.min().min(), X.max().max()
    ax.set_xlim([min, max])
    ax.set_ylim([min, max])
    ax.set_zlim([min, max])
    #ax.scatter(centers.iloc[:, 0], centers.iloc[:, 1], centers.iloc[:, 2], c='cyan', marker='*', s=100)

    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    file_name = 'KMeans_k_2'
    #fig.savefig('plots/' + file_name + '.png')
    #plt.close('all')
    print(metrics.silhouette_score(X, y_pred))
    plt.show()
    return X, metrics.silhouette_samples(X, y_pred)
    #return metrics.accuracy_score(y, y_pred), metrics.adjusted_mutual_info_score(y.iloc[:, 0].to_numpy(), y_pred), metrics.silhouette_score(X, y_pred), metrics.calinski_harabasz_score(X, y_pred), -1 * metrics.davies_bouldin_score(X, y_pred), file_name

results = pd.DataFrame()
i = 0
for features in [True]:
    for scaler in [True]:
        for outlier in [False]:
            """
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
            print("ami=%.2f, slihouette=%.2f, calinski=%.2f, davies_bouldin= %.2f, accuracy=%.2f" % (ami, silhouette, calinski, davies_bouldin, accuracy))
            results.to_csv('plots/automl_results_2.csv', index=False)
            """
            X, silhouette_samples = objective(features, scaler, outlier)
            X, silhouette_samples = pd.DataFrame(X), pd.DataFrame(silhouette_samples)
            X.to_csv('Xt.csv', index=False)
            silhouette_samples.to_csv('silhouette_samples.csv', index=False)

#results = results.sort_values(by=['silhouette', 'ami', 'accuracy'], ascending=False)
#results.to_csv('plots/fixed_pipeline_results.csv', index=False)

'''
plot silhouette chart
if config['metric'] == 'SIL':
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

    