import os
import json
import argparse

import pandas as pd

from os import listdir
from os.path import isfile, join
from matplotlib import gridspec


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=True, help="step of the pipeline to execute")
    parser.add_argument("-i", "--input", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input, args.output, args.pipeline

def create_directory(result_path, directory):
    result_path = os.path.join(result_path, directory)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path

def load_result(input_path, dataset, metric):
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

def save_result(results, output_path, dataset, metric):
    results.to_csv(os.path.join(output_path, dataset + '_' + metric + '_results.csv'), index=False)