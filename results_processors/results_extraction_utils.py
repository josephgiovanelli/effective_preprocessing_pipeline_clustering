import os
import json
import pandas as pd

from os import listdir
from os.path import isfile, join

from matplotlib import gridspec

metrics = ['SIL', 'CH', 'DBI']
datasets = ['avila', 'isolet', 'pendigits', 'postures', 'statlog']


def load_results(input_path):
    results = pd.DataFrame(columns=['dataset', 'metric', 'pipeline', 'algorithm', 'best_config', 'num_iterations', 'best_iteration', 'score', 'ami'])
    for dataset in datasets:
        for metric in metrics:
            for index in [0, 1]:
                try:
                    with open(os.path.join(input_path, dataset + '_' + metric + '_' + str(index) + '.json')) as json_file:
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
                    score = 0
                    ami = 0
                    best_config = ''
                    algorithm = ''
                    pipeline = ''
                    prototype = ''
                    num_iterations = 0
                    best_iteration = 0
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

def save_results(results, output_path):
    results.to_csv(os.path.join(output_path, 'union_results.csv'), index=False)
