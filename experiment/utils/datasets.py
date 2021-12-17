from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, fetch_covtype
import pandas as pd
import numpy as np
import os

def get_dataset(name):
    loader = {
        'breast': breast_cancer,
        'iris': iris,
        'wine': wine,
        'digits': digits,
        'covtype': covtype
        #'echr_article_1': echr.binary.get_dataset(article='1', flavors=[echr.Flavor.desc]).load
    }
    if name in loader:
        return loader[name]()
    else:
        return load_dataset(name)


def breast_cancer():
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names

def iris():
    data = load_iris()
    return data.data, data.target, data.feature_names

def wine():
    data = load_wine()
    return data.data, data.target, data.feature_names

def digits():
    digits = load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target, data.feature_names

def covtype():
    data = fetch_covtype
    return data.data, data.target, data.feature_names

def load_dataset(name):
    data = pd.read_csv(os.path.join('datasets', name +'.csv'), header=None)
    data = data.to_numpy()
    if name == 'parkinsons':
        features_name = [
            'MDVP:Fo(Hz)',
            'MDVP:Fhi(Hz)',
            'MDVP:Flo(Hz)',
            'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)',
            'MDVP:RAP',
            'MDVP:PPQ',
            'Jitter:DDP',
            'MDVP:Shimmer',
            'MDVP:Shimmer(dB)',
            'Shimmer:APQ3',
            'Shimmer:APQ5',
            'MDVP:APQ',
            'Shimmer:DDA',
            'NHR',
            'HNR',
            'RPDE',
            'DFA',
            'spread1',
            'spread2',
            'D2',
            'PPE'
            ]
    elif name == 'seeds':
        features_name = [
            'area',
            'perimeter',
            'compactness',
            'length of kernel',
            'width of kernel',
            'asymmetry coefficient',
            'length of kernel groove'
            ]
    elif name == 'synthetic_data' or name == 'synthetic':
        features_name = [
            'feature_0',
            'feature_1',
            'feature_2',
            'feature_3',
            'feature_4',
            'feature_5',
        ]
    else:
        raise Exception('No features names assigned')
    return data[:, :-1], data[:, -1], features_name
