import itertools
import os
import statistics

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn import metrics
from sklearn.manifold import TSNE

from matplotlib import cm

from scipy.spatial import distance

from utils import create_directory

def get_last_transformation(df, dataset, score_type, iteration):
    pipeline, is_there = {}, {}
    for transformation in ['features', 'normalize', 'outlier']:
        pipeline[transformation] = df.loc[(
            (df['dataset'] == dataset) & 
            (df['score_type'] == score_type) & 
            (df['iteration'] == iteration)), transformation].values[0]
        is_there[transformation] = pipeline[transformation] != 'None'
    last_transformation = 'outlier' if is_there['outlier'] else ('normalize' if is_there['normalize'] else ('features' if is_there['features'] else 'original'))
    return pipeline, last_transformation

def get_one_hot_encoding(current_features, original_features):
    features_to_return = []
    for feature in original_features:
        features_to_return.append(1 if feature in current_features else 0)
    return features_to_return

def diversificate(meta_features, conf):
    working_df = meta_features.copy()
    working_df = working_df.sort_values(by=['score'], ascending=False)
    first_solution = working_df.iloc[0]
    solutions = pd.DataFrame()
    solutions = solutions.append(first_solution)
    working_df = working_df.drop([first_solution.name])
    if conf['diversifaction_method'] == 'features_set' or conf['diversifaction_method'] == 'features_set_n_clusters':
        first_X = pd.read_csv(os.path.join(conf['diversification_input_path'], '_'.join([conf['dataset'], conf['score_type'], str(0), 'X', 'original']) + '.csv'))
        original_features = list(first_X.columns)

    for i in range(conf['num_results']-1):
        confs = working_df['iteration'].to_list()
        mmr = pd.DataFrame()
        for current_conf in confs:
            current_score = float(working_df[working_df['iteration'] == current_conf]['score'])
            if conf['diversifaction_method'] == 'clustering':
                current_y_pred = pd.read_csv(os.path.join(conf['diversification_input_path'], '_'.join([conf['dataset'], conf['score_type'], str(current_conf), 'y', 'pred']) + '.csv'))
                others = solutions['iteration'].to_list()
                distances = []
                for other in others:
                    other_y_pred = pd.read_csv(os.path.join(conf['diversification_input_path'], '_'.join([conf['dataset'], conf['score_type'], str(int(other)), 'y', 'pred']) + '.csv'))
                    ami = metrics.adjusted_mutual_info_score( current_y_pred.to_numpy().reshape(-1), other_y_pred.to_numpy().reshape(-1))
                    distances.append(1 - ami)
            else:
                _, current_last_transformation = get_last_transformation(meta_features, conf['dataset'], conf['score_type'], current_conf)
                current_X = pd.read_csv(os.path.join(conf['diversification_input_path'], '_'.join([conf['dataset'], conf['score_type'], str(current_conf), 'X', current_last_transformation]) + '.csv'))
                current_features = list(current_X.columns)
                current_features = get_one_hot_encoding(current_features, original_features)
                if conf['diversifaction_method'] == 'features_set_n_clusters':
                    current_features.append(meta_features.loc[(
                        (meta_features['dataset'] == conf['dataset']) & 
                        (meta_features['score_type'] == conf['score_type']) & 
                        (meta_features['iteration'] == current_conf)), 'algorithm__n_clusters'].values[0])
                others = solutions['iteration'].to_list()
                distances = []
                for other in others:
                    _, other_last_transformation = get_last_transformation(meta_features, conf['dataset'], conf['score_type'], int(other))
                    other_X = pd.read_csv(os.path.join(conf['diversification_input_path'], '_'.join([conf['dataset'], conf['score_type'], str(int(other)), 'X', other_last_transformation]) + '.csv'))
                    other_features = list(other_X.columns)
                    other_features = get_one_hot_encoding(other_features, original_features)
                    if conf['diversifaction_method'] == 'features_set_n_clusters':
                        other_features.append(meta_features.loc[(
                            (meta_features['dataset'] == conf['dataset']) & 
                            (meta_features['score_type'] == conf['score_type']) & 
                            (meta_features['iteration'] == int(other))), 'algorithm__n_clusters'].values[0])
                    if conf['distance_metric'] == 'euclidean':
                        dist = distance.euclidean(current_features, other_features)
                    else:
                        dist = distance.cosine(current_features, other_features)
                    distances.append(dist)
            current_mean_distance = statistics.mean(distances)
            current_mmr = (1 - conf['my_lambda']) * current_score + conf['my_lambda'] * current_mean_distance
            mmr = mmr.append({'iteration': current_conf, 'mmr': current_mmr}, ignore_index=True)
        mmr = mmr.sort_values(by=['mmr'], ascending=False)
        winning_conf = mmr.iloc[0]['iteration']
        winner = working_df.loc[working_df['iteration'] == winning_conf]
        solutions = solutions.append(winner)
        working_df = working_df.drop(winner.index)

def save_figure(solutions, conf):
    fig = plt.figure(figsize=(16, 9)) 
    i = 0  
    for _, row in solutions.iterrows():
        i += 1
        pipeline, last_transformation = get_last_transformation(solutions.copy(), conf['dataset'], conf['score_type'], int(row['iteration']))
        
        Xt = pd.read_csv(os.path.join(conf['diversification_input_path'], '_'.join([conf['dataset'], conf['score_type'], str(int(row['iteration'])), 'X', last_transformation]) + '.csv'))
        yt = pd.read_csv(os.path.join(conf['diversification_input_path'], '_'.join([conf['dataset'], conf['score_type'], str(int(row['iteration'])), 'y', 'pred']) + '.csv'))
        
        ax = fig.add_subplot(3, 3, i, projection='3d')
        colors = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'indigo', 'black'])
        if Xt.shape[1] > 3:
            Xt = pd.DataFrame(TSNE(n_components=3).fit_transform(Xt.to_numpy()), columns=['TSNE_0', 'TSNE_1', 'TSNE_2'])
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
        title = ' '.join([operator.split('_')[1] for operator in pipeline.values() if operator != 'None'])
        current_solution = solutions.loc[(
            (solutions['dataset'] == conf['dataset']) & 
            (solutions['score_type'] == conf['score_type']) & 
            (solutions['iteration'] == int(row['iteration']))), :]
        k_features = ' ' if pipeline['features'] == 'None' else ' k=' + str(current_solution.loc[:, 'features__k'].values[0]) + ' '
        n_clusters = 'n=' + str(int(current_solution.loc[:, 'algorithm__n_clusters'].values[0]))
        title += k_features + n_clusters
        ax.title.set_text(title)
    plt.tight_layout(pad=3.)
    fig.savefig(os.path.join(conf['diversification_output_path'], conf['output_file_name'] + '.pdf'))

def main():
    conf = {
        'dataset': 'breast',
        'score_type': 'sil',
        'num_results': 6,
        'my_lambda': 0.5,
        'diversifaction_method': 'features_set_n_clusters',
        'distance_metric': 'euclidean'
    }

    working_folder = conf['dataset'] + '_' + conf['score_type']
    path = os.path.join('results')
    diversification_path = os.path.join(path, 'clustering_diversification', working_folder)
    grid_search_path = os.path.join(path, 'grid_search', 'output')
    conf['diversification_input_path'] = os.path.join(diversification_path, 'input')
    conf['diversification_output_path'] = os.path.join(diversification_path, 'output')
    conf['output_file_name'] = 'diversification_output_' + '0_' + str(int(conf['my_lambda']*10)) + '_' + conf['diversifaction_method']
    if conf['diversifaction_method'] != 'clustering':
        conf['output_file_name'] += '_' + conf['distance_metric']

    meta_features = pd.read_csv(os.path.join(grid_search_path, 'grid_search_results.csv'))
    meta_features = meta_features[(meta_features['dataset'] == conf['dataset']) & (meta_features['score_type'] == conf['score_type'])]
    if conf['diversifaction_method'] == 'clustering':
        meta_features = meta_features[meta_features['outlier'] == 'None']
    
    try:
        solutions = pd.read_csv(os.path.join(conf['diversification_output_path'], conf['output_file_name'] + '.csv'))
    except:
        solutions = diversificate(meta_features, conf)
        solutions.to_csv(os.path.join(conf['diversification_output_path'], conf['output_file_name'] + '.csv'), index=False)
    
    save_figure(solutions, conf)

main()
