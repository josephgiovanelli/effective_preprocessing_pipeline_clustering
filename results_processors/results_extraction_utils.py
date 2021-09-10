import os
import json
import pandas as pd

from os import listdir
from os.path import isfile, join

from matplotlib import gridspec

metrics = ['SIL']
datasets = ['avila', 'isolet', 'pendigits', 'postures', 'statlog']


def load_results(input_path, only_best):
    results = pd.DataFrame(columns=['dataset', 'metric', 'pipeline', 'algorithm', 'best_config', 'num_iterations', 'best_iteration', 
                        'train_precision_weighted', 'test_precision_weighted', 'train_recall_weighted', 'test_recall_weighted', 
                        'train_f1_weighted', 'test_f1_weighted', 'train_accuracy', 'test_accuracy', 'train_adjusted_mutual_info_score', 
                        'test_adjusted_mutual_info_score', 'train_balanced_accuracy', 'test_balanced_accuracy'])
    for dataset in datasets:
        for metric in metrics:
            for index in [0, 1]:
                try:
                    success = True
                    best_suffix = 'best_pipeline_' if only_best else ''
                    file_name =  dataset + '_' + metric + '_' + best_suffix + str(index) + '.json'
                    with open(os.path.join(input_path, file_name)) as json_file:
                        data = json.load(json_file)
                        train_precision_weighted = data['context']['best_config']['train_precision_weighted']
                        test_precision_weighted = data['context']['best_config']['test_precision_weighted']
                        train_recall_weighted = data['context']['best_config']['train_recall_weighted']
                        test_recall_weighted = data['context']['best_config']['test_recall_weighted']
                        train_f1_weighted = data['context']['best_config']['train_f1_weighted']
                        test_f1_weighted = data['context']['best_config']['test_f1_weighted']
                        train_accuracy = data['context']['best_config']['train_accuracy']
                        test_accuracy = data['context']['best_config']['test_accuracy']
                        train_adjusted_mutual_info_score = data['context']['best_config']['train_adjusted_mutual_info_score']
                        test_adjusted_mutual_info_score = data['context']['best_config']['test_adjusted_mutual_info_score']
                        train_balanced_accuracy = data['context']['best_config']['train_balanced_accuracy']
                        score = data['context']['best_config']['score']
                        best_config = str(data['context']['best_config']).replace(' ', '').replace(',', ' ')
                        algorithm = str(data['context']['best_config']['algorithm']).replace(' ', '').replace(',', ' ')
                        pipeline = str(data['context']['best_config']['pipeline']).replace(' ', '').replace(',', ' ')
                        prototype = str(data['pipeline']).replace(' ', '')
                        num_iterations = data['context']['iteration'] + 1
                        best_iteration = data['context']['best_config']['iteration'] + 1
                except:
                    success = False
                    train_precision_weighted = 0
                    test_precision_weighted = 0
                    train_recall_weighted = 0
                    test_recall_weighted = 0
                    train_f1_weighted = 0
                    test_f1_weighted = 0
                    train_accuracy = 0
                    test_accuracy = 0
                    train_adjusted_mutual_info_score = 0
                    test_adjusted_mutual_info_score = 0
                    train_balanced_accuracy = 0
                    score = 0
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
                        'train_precision_weighted': [train_precision_weighted], 
                        'test_precision_weighted': [test_precision_weighted], 
                        'train_recall_weighted': [train_recall_weighted], 
                        'test_recall_weighted': [test_recall_weighted], 
                        'train_f1_weighted': [train_f1_weighted], 
                        'test_f1_weighted': [test_f1_weighted], 
                        'train_accuracy': [train_accuracy], 
                        'test_accuracy': [test_accuracy], 
                        'train_adjusted_mutual_info_score': [train_adjusted_mutual_info_score], 
                        'test_adjusted_mutual_info_score': [test_adjusted_mutual_info_score], 
                        'train_balanced_accuracy': [train_balanced_accuracy], 
                        'test_balanced_accuracy': [score],
                    }), ignore_index=True)

    return results

def save_results(results, output_path, only_best):
    results.to_csv(os.path.join(output_path, 'union_results' + ('_only_best' if only_best else '') + '.csv'), index=False)
