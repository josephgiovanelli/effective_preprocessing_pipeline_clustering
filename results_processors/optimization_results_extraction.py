from __future__ import print_function

import itertools
import os 

import pandas as pd

from utils import load_result, save_result
from utils import create_directory


def main():
    optimization_approaches = ['exhaustive', 'smbo']
    for approach in optimization_approaches:
        input_path = os.path.join('results', 'optimization', approach)
        output_path = create_directory(input_path, 'summary')
        output_file_name = 'summary.csv'
        results = pd.DataFrame()
        for dataset in ['iris', 'wine', 'breast', 'seeds', 'parkinsons', 'synthetic_data']:
            for metric in ['sil', 'ch', 'dbi']:
                try:
                    results = results.append(load_result(input_path, dataset=dataset, metric=metric))
                except Exception as e:
                    print(e)
        results.to_csv(os.path.join(output_path, output_file_name), index=False)

main()