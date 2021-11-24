from __future__ import print_function

import itertools
import os 

import pandas as pd

from utils import load_result, save_result
from utils import create_directory


def main():
    path = os.path.join('results', 'grid_search')
    input_path = os.path.join(path, 'input')
    output_path = os.path.join(path, 'output')
    output_file_name = 'grid_search_results.csv'
    results = pd.DataFrame()
    for dataset in ['iris', 'wine', 'breast', 'seeds', 'parkinsons', 'synthetic_data']:
        for metric in ['sil', 'ch', 'dbi']:
            results = results.append(load_result(input_path, dataset=dataset, metric=metric))
    results.to_csv(os.path.join(output_path, output_file_name), index=False)

main()