import os
import sys
import statistics
import yaml
import warnings
import itertools
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.manifold import TSNE
from matplotlib import cm
from scipy.spatial import distance
from utils import create_directory
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..' )
sys.path.append( mymodule_dir )
from experiment.pipeline.outlier_detectors import MyOutlierDetector
from experiment.utils import datasets

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

def diversificate_mmr(meta_features, conf, original_features):
    working_df = meta_features.copy()
    working_df = working_df.sort_values(by=['score'], ascending=False)
    first_solution = working_df.iloc[0]
    solutions = pd.DataFrame()
    solutions = solutions.append(first_solution)
    working_df = working_df.drop([first_solution.name])

    for i in range(conf['num_results']-1):
        confs = working_df['iteration'].to_list()
        mmr = pd.DataFrame()
        for current_conf in confs:
            current_score = float(working_df[working_df['iteration'] == current_conf]['score'])
            if conf['diversifaction_distance'] == 'clustering':
                current_y_pred = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['score_type'], str(current_conf), 'y', 'pred']) + '.csv'))
                others = solutions['iteration'].to_list()
                distances = []
                for other in others:
                    other_y_pred = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['score_type'], str(int(other)), 'y', 'pred']) + '.csv'))
                    ami = metrics.adjusted_mutual_info_score( current_y_pred.to_numpy().reshape(-1), other_y_pred.to_numpy().reshape(-1))
                    distances.append(1 - ami)
            else:
                _, current_last_transformation = get_last_transformation(meta_features, conf['dataset'], conf['score_type'], current_conf)
                current_X = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['score_type'], str(current_conf), 'X', current_last_transformation]) + '.csv'))
                current_features = list(current_X.columns)
                current_features = get_one_hot_encoding(current_features, original_features)
                if conf['diversifaction_distance'] == 'features_set_n_clusters':
                    current_features.append(meta_features.loc[(
                        (meta_features['dataset'] == conf['dataset']) & 
                        (meta_features['score_type'] == conf['score_type']) & 
                        (meta_features['iteration'] == current_conf)), 'algorithm__n_clusters'].values[0])
                others = solutions['iteration'].to_list()
                distances = []
                for other in others:
                    _, other_last_transformation = get_last_transformation(meta_features, conf['dataset'], conf['score_type'], int(other))
                    other_X = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['score_type'], str(int(other)), 'X', other_last_transformation]) + '.csv'))
                    other_features = list(other_X.columns)
                    other_features = get_one_hot_encoding(other_features, original_features)
                    if conf['diversifaction_distance'] == 'features_set_n_clusters':
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
            current_mmr = (1 - conf['lambda']) * current_score + conf['lambda'] * current_mean_distance
            mmr = mmr.append({'iteration': current_conf, 'mmr': current_mmr}, ignore_index=True)
        mmr = mmr.sort_values(by=['mmr'], ascending=False)
        winning_conf = mmr.iloc[0]['iteration']
        winner = working_df.loc[working_df['iteration'] == winning_conf]
        solutions = solutions.append(winner)
        working_df = working_df.drop(winner.index)
    dashboard_score = evaluate_dashboard(solutions.copy(), conf, original_features)
    return {'solutions': solutions, 'score': dashboard_score}

def diversificate_exhaustive(meta_features, conf, original_features):
    working_df = meta_features.copy()
    cc = list(itertools.combinations(list(working_df.index), conf['num_results']))
    exhaustive_search = pd.DataFrame()
    best_dashboard = {'solutions': pd.DataFrame(), 'score': 0.}
    for c in cc:
        solutions = working_df.loc[c, :].copy()
        dashboard_score = evaluate_dashboard(solutions, conf, original_features)
        if dashboard_score > best_dashboard['score']:
            best_dashboard['solutions'] = solutions.copy()
            best_dashboard['score'] = dashboard_score
        exhaustive_search = exhaustive_search.append({
            'conf_0': int(solutions.loc[c[0], 'iteration']),
            'conf_1': int(solutions.loc[c[1], 'iteration']),
            'conf_2': int(solutions.loc[c[2], 'iteration']),
            'conf_3': int(solutions.loc[c[3], 'iteration']),
            'conf_4': int(solutions.loc[c[4], 'iteration']),
            'conf_5': int(solutions.loc[c[5], 'iteration']),
            'dashboard_score': dashboard_score,
        }, ignore_index=True)
    exhaustive_search.sort_values(by=['mmr'], ascending=False)
    exhaustive_search.to_csv(os.path.join(conf['output_path'], conf['output_file_name'] + '_all' + '.csv'), index=False)
    return best_dashboard

def evaluate_dashboard(solutions, conf, original_features):
    def compute_pairwise_div(df, conf, original_features):
        df = df.reset_index()
        first_iteration = int(df.loc[0, 'iteration'])
        second_iteration = int(df.loc[1, 'iteration'])
        div_vectors = []
        for iteration in [first_iteration, second_iteration]:
            if conf['diversifaction_distance'] == 'clustering':
                y_pred = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['score_type'], str(iteration), 'y', 'pred']) + '.csv'))
                div_vectors.append(y_pred)
            else:
                _, last_transformation = get_last_transformation(df, conf['dataset'], conf['score_type'], iteration)
                X = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['score_type'], str(iteration), 'X', last_transformation]) + '.csv'))
                features = list(X.columns)
                features = get_one_hot_encoding(features, original_features)
                if conf['diversifaction_distance'] == 'features_set_n_clusters':
                    features.append(df.loc[df['iteration'] == iteration, 'algorithm__n_clusters'])
                div_vectors.append(features)
        if conf['diversifaction_distance'] == 'clustering':
            return metrics.adjusted_mutual_info_score(div_vectors[0].to_numpy().reshape(-1), div_vectors[1].to_numpy().reshape(-1))
        else:
            if conf['distance_metric'] == 'euclidean':
                return distance.euclidean(div_vectors[0], div_vectors[1])
            else:
                return distance.cosine(div_vectors[0], div_vectors[1])

    sim = solutions['score'].sum()
    cc = list(itertools.combinations(list(solutions.index), 2))
    div = sum([compute_pairwise_div(solutions.loc[c, :].copy(), conf, original_features) for c in cc])
    return ((conf['num_results'] - 1) * (1 - conf['lambda']) * sim) + (2 * conf['lambda'] * div)

def save_figure(solutions, conf):
    fig = plt.figure(figsize=(32, 18)) 
    i = 0  
    for _, row in solutions.iterrows():
        i += 1
        pipeline, last_transformation = get_last_transformation(solutions.copy(), conf['dataset'], conf['score_type'], int(row['iteration']))
        
        Xt = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['score_type'], str(int(row['iteration'])), 'X', last_transformation]) + '.csv'))
        yt = pd.read_csv(os.path.join(conf['input_path'], '_'.join([conf['dataset'], conf['score_type'], str(int(row['iteration'])), 'y', 'pred']) + '.csv'))
        
        if Xt.shape[1] < 3:
            ax = fig.add_subplot(3, 3, i)
        else:
            ax = fig.add_subplot(3, 3, i, projection='3d')
        colors = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'indigo', 'black'])
        old_X = Xt.copy()
        if Xt.shape[1] > 3:
            Xt = pd.DataFrame(TSNE(n_components=3, random_state=42).fit_transform(Xt.to_numpy(), yt.to_numpy()), columns=['TSNE_0', 'TSNE_1', 'TSNE_2'])
            if conf['outlier']:
                Xt, yt = MyOutlierDetector(n_neighbors=32).fit_resample(Xt.to_numpy(), yt.iloc[:, 0].to_numpy())
                Xt, yt = pd.DataFrame(Xt, columns=['TSNE_0', 'TSNE_1', 'TSNE_2']), pd.DataFrame(yt, columns=['target'])
        n_selected_features = Xt.shape[1]
        Xt = Xt.iloc[:, :n_selected_features]
        min, max = Xt.min().min(), Xt.max().max()
        range = (max-min)/10
        xs = Xt.iloc[:, 0]
        ys = [(max+min)/2] * Xt.shape[0] if n_selected_features < 2 else Xt.iloc[:, 1]
        zs = [(max+min)/2] * Xt.shape[0] if n_selected_features < 3 else Xt.iloc[:, 2]
        if Xt.shape[1] < 3:
            ax.scatter(xs, ys, c=[colors[int(i)] for i in yt.iloc[:, 0].to_numpy()])
        else:
            ax.scatter(xs, ys, zs, c=[colors[int(i)] for i in yt.iloc[:, 0].to_numpy()])
        ax.set_xlim([min - range, max + range])
        ax.set_ylim([min - range, max + range])
        ax.set_xlabel(list(Xt.columns)[0], fontsize=16)
        ax.set_ylabel('None' if n_selected_features < 2 else list(Xt.columns)[1], fontsize=16)
        if Xt.shape[1] >= 3:
            ax.set_zlim([min, max])
            ax.set_zlabel('None' if n_selected_features < 3 else list(Xt.columns)[2], fontsize=16)
        title = '\n'.join([operator for operator in pipeline.values() ])
        current_solution = solutions.loc[(
            (solutions['dataset'] == conf['dataset']) & 
            (solutions['score_type'] == conf['score_type']) & 
            (solutions['iteration'] == int(row['iteration']))), :]
        k_features = '\nk= ' + str(old_X.shape[1])
        n_clusters = '    n=' + str(int(current_solution.loc[:, 'algorithm__n_clusters'].values[0]))
        title += k_features + n_clusters
        title += '\nscore=' + str(round(current_solution.loc[:, 'score'].values[0], 2))
        title += '    ami=' + str(round(current_solution.loc[:, 'ami'].values[0], 2))
        ax.set_title(title, fontdict={'fontsize': 20, 'fontweight': 'medium'})
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    title = f'''dataset: {conf['dataset']}, score_type: {conf['score_type']}, num_results: {conf['num_results']}, lambda: {conf['lambda']}, diversifaction_distance: {conf['diversifaction_distance']}'''
    title += '' if 'distance_metric' not in conf else f''', distance_metric: {conf['distance_metric']}'''
    fig.suptitle(title, fontsize=30)
    fig.savefig(os.path.join(conf['output_path'], conf['output_file_name'] + ('_outlier' if conf['outlier'] else '') + '.pdf'))

def main():
    path = os.path.join('results')
    diversification_path = create_directory(path, 'diversification')

    with open(os.path.join(diversification_path, 'conf.yaml'), 'r') as stream:
        try:
            confs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    

    for i in range(len(confs)):
        conf = confs[i]
        print(f'''{i+1}th conf out of {len(confs)}: {conf}''')

        working_folder = conf['dataset'] + '_' + conf['score_type']
        conf['diversification_path'] = os.path.join(diversification_path, 'opt_' + conf['optimization_method'])
        conf['output_path'] = create_directory(conf['diversification_path'], working_folder)
        conf['output_file_name'] = 'mmr_' + '0_' + str(int(conf['lambda']*10)) + '_' + conf['diversifaction_distance']
        optimization_path = os.path.join(path, 'optimization', conf['optimization_method'])
        conf['input_path'] = os.path.join(optimization_path, 'details', working_folder)

        
        meta_features = pd.read_csv(os.path.join(optimization_path, 'summary', 'summary.csv'))
        meta_features = meta_features[(meta_features['dataset'] == conf['dataset']) & (meta_features['score_type'] == conf['score_type'])]
        if conf['diversifaction_distance'] == 'clustering':
            meta_features = meta_features[meta_features['outlier'] == 'None']
        meta_features = meta_features[(meta_features['normalize__with_mean'] == 'None') | (meta_features['normalize__with_mean'] == 'True')]
        meta_features = meta_features[(meta_features['normalize__with_std'] == 'None') | (meta_features['normalize__with_std'] == 'True')]

        meta_features1 = meta_features[meta_features['features__k'] == 'None']
        meta_features2 = meta_features[meta_features['features__k'] != 'None']
        _, _, original_features = datasets.get_dataset(conf['dataset'])
        meta_features2 = meta_features2[meta_features2['features__k'].astype(np.int32) < len(original_features)]
        meta_features = pd.concat([meta_features1, meta_features2], ignore_index=True)

        #meta_features = meta_features[meta_features['score'] >= 0.4]
        meta_features = meta_features[~((meta_features['normalize'] != 'None') & (meta_features['features__k'] == '1'))]
        
        
        conf['output_file_name'] = conf['diversification_method'] + '_0_' + str(int(conf['lambda']*10)) + '_' + conf['diversifaction_distance']
        if conf['diversifaction_distance'] != 'clustering':
            conf['output_file_name'] += '_' + conf['distance_metric']
        try:
            dashboard = {}
            dashboard['solutions'] = pd.read_csv(os.path.join(conf['output_path'], conf['output_file_name'] + '.csv'))
            print('        A previous diversification result was found')
            print('        Calculating score')
            dashboard['score'] = evaluate_dashboard(dashboard['solutions'], conf, original_features)
        except:
            print('        A previous diversification result was not found')
            print('        Diversification process starts')
            if conf['diversification_method'] == 'mmr':
                dashboard = diversificate_mmr(meta_features, conf, original_features)
            else:
                dashboard = diversificate_exhaustive(meta_features, conf, original_features)
            print('        Diversification process ends')
            dashboard['solutions'].to_csv(os.path.join(conf['output_path'], conf['output_file_name'] + '.csv'), index=False)
        dashboard_score = dashboard['score']
        print(f'        Dashboard score: {dashboard_score}')

        print('        Plotting')
        for outlier_removal in [True, False]:
            conf['outlier'] = outlier_removal
            plot_path = os.path.join(conf['output_path'], conf['output_file_name'] + ('_outlier' if conf['outlier'] else '') + '.pdf')
            if not os.path.exists(plot_path):
                save_figure(dashboard['solutions'], conf)     
        
main()