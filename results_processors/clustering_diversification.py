from __future__ import print_function

import itertools
import os
import statistics

import pandas as pd

from sklearn import metrics

from results_extraction_utils import load_just_one_result, save_just_one_result
from utils import create_directory


def main():
    meta_input_path = os.path.join('results')
    output_path = os.path.join('results')
    dataset = 'breast'
    score_type = 'sil'
    clustering_input_path = os.path.join(
        'results', 'clustering_diversification', dataset + '_' + score_type)
    num_results = 5
    my_lambda = 0.5

    df = pd.read_csv(os.path.join(meta_input_path, 'diversification_input.csv'))
    df = df[['dataset', 'score_type', 'iteration', 'score']]
    df = df[(df['dataset'] == dataset) & (df['score_type'] == score_type)]
    df = df.sort_values(by=['score'], ascending=False)
    # print(df)

    first_solution = df.iloc[0]
    solutions = pd.DataFrame()
    solutions = solutions.append(first_solution)
    df = df.drop([first_solution.name])
    
    for i in range(num_results-1):
        confs = df['iteration'].to_list()
        mmr = pd.DataFrame()
        for current_conf in confs:
            current_score = float(df[df['iteration'] == current_conf]['score'])
            current_y_pred = pd.read_csv(os.path.join(clustering_input_path, '_'.join(
                [dataset, score_type, str(current_conf), 'y', 'pred']) + '.csv'), header=None)
            others = solutions['iteration'].to_list()
            distances = []
            for other in others:
                other_y_pred = pd.read_csv(os.path.join(clustering_input_path, '_'.join(
                    [dataset, score_type, str(int(other)), 'y', 'pred']) + '.csv'), header=None)
                ami = metrics.adjusted_mutual_info_score(
                    current_y_pred.to_numpy().reshape(-1), other_y_pred.to_numpy().reshape(-1))
                distances.append(1 - ami)
            current_mean_distance = statistics.mean(distances)
            current_mmr = (1 - my_lambda) * current_score + my_lambda * current_mean_distance
            mmr = mmr.append({'iteration': current_conf, 'mmr': current_mmr}, ignore_index=True)
        mmr = mmr.sort_values(by=['mmr'], ascending=False)
        winning_conf = mmr.iloc[0]['iteration']
        winner = df.loc[df['iteration'] == winning_conf]
        solutions = solutions.append(winner)
        df = df.drop(winner.index)
    solutions.to_csv(os.path.join(output_path, 'diversification_output.csv'), index=False)


main()
