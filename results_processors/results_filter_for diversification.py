from __future__ import print_function

import itertools
import os 

import pandas as pd

from results_extraction_utils import load_just_one_result, save_just_one_result
from utils import create_directory


def main():
    input_path = os.path.join('results')
    output_path = os.path.join('results')
    results = pd.read_csv(os.path.join(input_path, 'grid_search_results.csv'))
    results = results[results['outlier'] == 'None']
    results.to_csv(os.path.join(output_path, 'diversification_input.csv'), index=False)
main()