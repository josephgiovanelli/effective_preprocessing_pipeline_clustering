from __future__ import print_function

import itertools
import os
import statistics

import pandas as pd

from sklearn import metrics

from results_extraction_utils import load_just_one_result, save_just_one_result
from utils import create_directory

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def main():
    meta_input_path = os.path.join('results')
    output_path = os.path.join('results')
    dataset = 'breast'
    score_type = 'sil'
    clustering_input_path = os.path.join('results', 'clustering_diversification', dataset + '_' + score_type)
    num_results = 6
    my_lambda = 0.7
    apply_diversification = True

    df = pd.read_csv(os.path.join(meta_input_path, 'grid_search_results.csv'))
    df = df[df['outlier'] == 'None']
    meta_features = df.copy()
    #df = df[['dataset', 'score_type', 'iteration', 'score']]
    df = df[(df['dataset'] == dataset) & (df['score_type'] == score_type)]
    df = df.sort_values(by=['score'], ascending=False)

    if apply_diversification:
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
                    [dataset, score_type, str(current_conf), 'y', 'pred']) + '.csv'))
                others = solutions['iteration'].to_list()
                distances = []
                for other in others:
                    other_y_pred = pd.read_csv(os.path.join(clustering_input_path, '_'.join([dataset, score_type, str(int(other)), 'y', 'pred']) + '.csv'))
                    ami = metrics.adjusted_mutual_info_score( current_y_pred.to_numpy().reshape(-1), other_y_pred.to_numpy().reshape(-1))
                    distances.append(1 - ami)
                current_mean_distance = statistics.mean(distances)
                current_mmr = (1 - my_lambda) * current_score + my_lambda * current_mean_distance
                mmr = mmr.append({'iteration': current_conf, 'mmr': current_mmr}, ignore_index=True)
            mmr = mmr.sort_values(by=['mmr'], ascending=False)
            winning_conf = mmr.iloc[0]['iteration']
            winner = df.loc[df['iteration'] == winning_conf]
            solutions = solutions.append(winner)
            df = df.drop(winner.index)

        for index, row in solutions.iterrows():
            for transformation in ['normalize', 'features', 'outlier']:
                solutions.loc[solutions['iteration'] == row['iteration'], transformation] = meta_features.loc[
                    ((meta_features['dataset'] == dataset) &
                    (meta_features['score_type'] == score_type) &
                    (meta_features['iteration'] == row['iteration'])), transformation] != 'None'
        
        solutions.to_csv(os.path.join(output_path, 'diversification_output.csv'), index=False)
    else:
        solutions = pd.read_csv(os.path.join(output_path, 'diversification_output.csv'))

    
    fig = plt.figure(figsize=(15,12)) 
    i = 0  
    for index, row in solutions.iterrows():
        i += 1
        is_there = {}
        for transformation in ['normalize', 'features', 'outlier']:
            is_there[transformation] = solutions.loc[(
                (solutions['dataset'] == dataset) & 
                (solutions['score_type'] == score_type) & 
                (solutions['iteration'] == row['iteration'])), transformation].values[0]
        last_transformation = 'outlier' if is_there['outlier'] else ('normalize' if is_there['normalize'] else ('features' if is_there['features'] else 'original'))
        Xt = pd.read_csv(os.path.join(clustering_input_path, '_'.join([dataset, score_type, str(int(row['iteration'])), 'X', last_transformation]) + '.csv'))
        yt = pd.read_csv(os.path.join(clustering_input_path, '_'.join([dataset, score_type, str(int(row['iteration'])), 'y', 'pred']) + '.csv'))
        
        ax = fig.add_subplot(3, 3, i, projection='3d')
        colors = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'indigo', 'black'])
        n_selected_features = Xt.shape[1] if Xt.shape[1] < 3 else 3
        Xt = Xt.iloc[:, :n_selected_features]
        min, max = Xt.min().min(), Xt.max().max()
        xs = Xt.iloc[:, 0]
        ys = [min] * Xt.shape[0] if n_selected_features < 2 else Xt.iloc[:, 1]
        zs = [min] * Xt.shape[0] if n_selected_features < 3 else Xt.iloc[:, 2]
        ax.scatter(xs, ys,  zs, c=[colors[int(i)] for i in yt.iloc[:, 0].to_numpy()])
        ax.set_xlim([min, max])
        ax.set_ylim([min, max])
        ax.set_zlim([min, max])
        ax.set_xlabel(list(Xt.columns)[0])
        ax.set_ylabel('None' if n_selected_features < 2 else list(Xt.columns)[1])
        ax.set_zlabel('None' if n_selected_features < 3 else list(Xt.columns)[2])

    
    fig.savefig(os.path.join(output_path, 'diversification_output.pdf'))


main()
