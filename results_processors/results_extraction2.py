from __future__ import print_function

import itertools
import os 

import pandas as pd

from results_extraction_utils import load_just_one_result, save_just_one_result
from utils import create_directory


def main():
    input_path = os.path.join('results', 'silhouette_study')
    output_path = os.path.join('results')
    results = pd.DataFrame()
    for dataset in ['iris', 'wine', 'breast', 'seeds', 'parkinsons', 'synthetic_data']:
        results = results.append(load_just_one_result(input_path, dataset=dataset, metric='sil'))
    results.to_csv(os.path.join(output_path, 'grid_search_results.csv'), index=False)

main()