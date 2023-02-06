import os
import json
import argparse

import pandas as pd

from os import listdir
from os.path import isfile, join
from matplotlib import gridspec

def load_result(input_path, dataset, metric):
    results = pd.DataFrame()
    file_name =  dataset + '_' + metric + '.json'
    with open(os.path.join(input_path, file_name)) as json_file:
        data = json.load(json_file)
        history = data['context']['history']
        for elem in history:
            results = results.append(pd.DataFrame({
                'dataset': [dataset],
                'iteration': [elem['iteration']], 
                #'pipeline': [elem['pipeline']], 
                #'algorithm': [elem['algorithm']], 
                'features': ['None' if elem['pipeline']['features'][0] == 'features_NoneType' else elem['pipeline']['features'][0]], 
                'features__k': ['None' if elem['pipeline']['features'][0] == 'features_NoneType' else elem['pipeline']['features'][1]['features__k']], 
                'normalize': ['None' if elem['pipeline']['normalize'][0] == 'normalize_NoneType' else elem['pipeline']['normalize'][0]], 
                #'normalize__with_mean': ['None' if (elem['pipeline']['normalize'][0] == 'normalize_NoneType' or elem['pipeline']['normalize'][0] == 'normalize_MinMaxScaler') else elem['pipeline']['normalize'][1]['normalize__with_mean']], 
                #'normalize__with_std': ['None' if (elem['pipeline']['normalize'][0] == 'normalize_NoneType' or elem['pipeline']['normalize'][0] == 'normalize_MinMaxScaler') else elem['pipeline']['normalize'][1]['normalize__with_std']], 
                'outlier': ['None' if elem['pipeline']['outlier'][0] == 'outlier_NoneType' else elem['pipeline']['outlier'][0]], 
                'outlier__n_neighbors': ['None' if elem['pipeline']['outlier'][0] == 'outlier_NoneType' or elem['pipeline']['outlier'][0] =='outlier_IsolationOutlierDetector' else elem['pipeline']['outlier'][1]['outlier__n_neighbors']], 
                'algorithm': [elem['algorithm'][0]], 
                #'algorithm__max_iter': [elem['algorithm'][1]['max_iter']], 
                'algorithm__n_clusters': [elem['algorithm'][1]['n_clusters']], 
                'optimization_internal_metric': [metric], 
                'optimization_external_metric': ['ami'], 
                'optimization_internal_metric_value': [elem['internal_metric']], 
                'optimization_external_metric_value': [elem['external_metric']],
                'max_optimization_internal_metric_value': [elem['max_history_internal_metric']], 
                'max_optimization_external_metric_value': [elem['max_history_external_metric']],
                'duration': [elem['duration']]
            }), ignore_index=True)

    return results

def save_result(results, output_path, dataset, metric):
    results.to_csv(os.path.join(output_path, dataset + '_' + metric + '_results.csv'), index=False)