import json
import os
from results_processors.results_processors_utils import get_dataset

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from pipeline.feature_selectors import PearsonThreshold
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    Birch,
)
from sklearn.mixture import GaussianMixture


# from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour
# from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    KBinsDiscretizer,
    Binarizer,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from pipeline.feature_selectors import PearsonThreshold

from imblearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion

from pipeline.PrototypeSingleton import PrototypeSingleton
from pipeline.outlier_detectors import (
    LocalOutlierDetector,
    IsolationOutlierDetector,
)  # , SGDOutlierDetector

from fsfc.generic import GenericSPEC, NormalizedCut, WKMeans

from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


colors = np.array(
    [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "grey",
        "olive",
        "cyan",
        "indigo",
        "black",
    ]
)


def plot_cluster_data(df, target_column):
    """
    Plot a dataframe with clustering results using a scatter plot.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to be plotted.
    target_column (str): The name of the column that contains the cluster labels.

    Returns:
    None
    """

    # Make sure the target_column is in the dataframe
    if target_column not in df.columns:
        raise ValueError(f"'{target_column}' column not found in the dataframe.")

    # Create a scatter plot for each cluster
    unique_clusters = df[target_column].unique()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    df.columns = [
        "[0 - mcg (numeric)]",
        "[1 - gvh (numeric)]",
        "[2 - lip (numeric)]",
        "[3 - chg (numeric)]",
        "[4 - aac (numeric)]",
        "[5 - alm1 (numeric)]",
        "target",
    ]
    print(df.iloc[:, :-1])
    Xt = pd.concat(
        [
            pd.DataFrame(
                PCA(n_components=2, random_state=42).fit_transform(
                    # pd.read_csv(
                    #     "results/optimization/smbo/details/ecoli_sil/ecoli_sil_478_X_normalize.csv"
                    # ).to_numpy(),
                    df.iloc[:, :-1].to_numpy(),
                    df.iloc[:, -1].to_numpy(),
                ),
                columns=["TSNE_0", "TSNE_1"],
            ),
            df[target_column],
        ],
        axis=1,
    )
    # print(Xt)

    for i, cluster_label in enumerate(unique_clusters):
        print(i)
        cluster_data = Xt[Xt[target_column] == cluster_label]
        plt.scatter(
            cluster_data.iloc[:, 0],
            cluster_data.iloc[:, 1],
            c=[colors[i]] * cluster_data.shape[0],
            label=f"Cluster {cluster_label}",
        )

        n_selected_features = Xt.shape[1]
        Xt = Xt.iloc[:, :n_selected_features]
        min, max = Xt.min().min(), Xt.max().max()
        range = (max - min) / 10
        xs = Xt.iloc[:, 0]
        ys = (
            [(max + min) / 2] * Xt.shape[0]
            if n_selected_features < 2
            else Xt.iloc[:, 1]
        )
        # zs = [(max+min)/2] * Xt.shape[0] if n_selected_features < 3 else Xt.iloc[:, 2]
        if Xt.shape[1] < 3:
            ax.scatter(xs, ys, c=[colors[int(i)] for i in yt.iloc[:, 0].to_numpy()])
        # else:
        # ax.scatter(xs, ys, zs, c=[colors[int(i)] for i in yt.iloc[:, 0].to_numpy()])
        ax.set_xlim([min - range, max + range])
        ax.set_ylim([min - range, max + range])
        ax.set_xlabel(list(Xt.columns)[0], fontsize=16)
        ax.set_ylabel(
            "None" if n_selected_features < 2 else list(Xt.columns)[1], fontsize=16
        )
        # if Xt.shape[1] >= 3:
        # ax.set_zlim([min, max])
        # ax.set_zlabel('None' if n_selected_features < 3 else list(Xt.columns)[2], fontsize=16)

    return fig


if __name__ == "__main__":
    X, y, _ = get_dataset("ecoli")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    pipe = Pipeline(
        [
            ("features", PearsonThreshold(threshold=0.7)),
            ("scaler", StandardScaler()),
            ("svc", AgglomerativeClustering(n_clusters=3, linkage="complete")),
            # ("svc", KMeans(n_clusters=2)),
        ]
    )
    Xt = pipe[:-1].fit_transform(X)
    pd.DataFrame(Xt).to_csv("clusering.csv", index=False)
    y_pred = pipe.fit_predict(X, y)
    internal_metric = silhouette_score(Xt, y_pred)
    print(len([elem for elem in y_pred if elem == 1]))
    df = pd.concat([pd.DataFrame(Xt), pd.DataFrame(y_pred, columns=["target"])], axis=1)

    fig = plot_cluster_data(df, "target")

    # Save the figure as a PNG file
    fig.savefig("clustering_plot.png", dpi=300, bbox_inches="tight")

    # Close the figure (optional)
    plt.close(fig)

    print(internal_metric)
