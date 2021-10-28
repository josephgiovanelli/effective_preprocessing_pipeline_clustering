import os
import json
import pandas as pd

from os import listdir
from os.path import isfile, join

from matplotlib import gridspec

metrics = ['SIL', 'CH', 'DBI']
datasets = ['synthetic_data']


def load_results(input_path, only_best):
    results = pd.DataFrame(columns=['dataset', 'metric', 'pipeline', 'algorithm', 'best_config', 'num_iterations', 'best_iteration', 'score', 'ami'])
    for dataset in datasets:
        for metric in metrics:
            for index in [0, 1]:
                try:
                    success = True
                    best_suffix = 'best_pipeline_' if only_best else ''
                    file_name =  dataset + '_' + metric + '_' + best_suffix + str(index) + '.json'
                    with open(os.path.join(input_path, file_name)) as json_file:
                        data = json.load(json_file)
                        score = data['context']['best_config']['score']
                        ami = data['context']['best_config']['ami']
                        best_config = str(data['context']['best_config']).replace(' ', '').replace(',', ' ')
                        algorithm = str(data['context']['best_config']['algorithm']).replace(' ', '').replace(',', ' ')
                        pipeline = str(data['context']['best_config']['pipeline']).replace(' ', '').replace(',', ' ')
                        prototype = str(data['pipeline']).replace(' ', '')
                        num_iterations = data['context']['iteration'] + 1
                        best_iteration = data['context']['best_config']['iteration'] + 1
                except:
                    success = False
                    score = 0
                    ami = 0
                    best_config = ''
                    algorithm = ''
                    pipeline = ''
                    prototype = ''
                    num_iterations = 0
                    best_iteration = 0

                if not(only_best) or success:
                    results = results.append(pd.DataFrame({
                        'dataset': [dataset], 
                        'metric': [metric], 
                        'pipeline': [prototype], 
                        'algorithm': [algorithm], 
                        'best_config': [best_config], 
                        'num_iterations': [num_iterations], 
                        'best_iteration': [best_iteration], 
                        'score': [score], 
                        'ami': [ami]
                    }), ignore_index=True)

    return results

def save_results(results, output_path, only_best):
    results.to_csv(os.path.join(output_path, 'union_results' + ('_only_best' if only_best else '') + '.csv'), index=False)

def load_just_one_result(input_path, dataset, metric):
    results = pd.DataFrame()
    file_name =  dataset + '_' + metric.lower() + '_best_pipeline_0.json'
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
                'normalize__with_mean': ['None' if (elem['pipeline']['normalize'][0] == 'normalize_NoneType' or elem['pipeline']['normalize'][0] == 'normalize_MinMaxScaler') else elem['pipeline']['normalize'][1]['normalize__with_mean']], 
                'normalize__with_std': ['None' if (elem['pipeline']['normalize'][0] == 'normalize_NoneType' or elem['pipeline']['normalize'][0] == 'normalize_MinMaxScaler') else elem['pipeline']['normalize'][1]['normalize__with_std']], 
                'outlier': ['None' if elem['pipeline']['outlier'][0] == 'outlier_NoneType' else elem['pipeline']['outlier'][0]], 
                'outlier__n_neighbors': ['None' if elem['pipeline']['outlier'][0] == 'outlier_NoneType' else elem['pipeline']['outlier'][1]['outlier__n_neighbors']], 
                'algorithm': [elem['algorithm'][0]], 
                'algorithm__max_iter': [elem['algorithm'][1]['max_iter']], 
                'algorithm__n_clusters': [elem['algorithm'][1]['n_clusters']], 
                'score_type': [metric], 
                'score': [elem['score']], 
                'ami': [elem['ami']],
                'max_history_score': [elem['max_history_score']], 
                'max_history_score_ami': [elem['max_history_score_ami']]
            }), ignore_index=True)

    return results

def save_just_one_result(results, output_path, dataset, metric):
    results.to_csv(os.path.join(output_path, dataset + '_' + metric + '_results.csv'), index=False)