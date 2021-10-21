from __future__ import print_function

import itertools
import os 

from results_extraction_utils import load_just_one_result, save_just_one_result
from utils import create_directory


def main():
    input_path = os.path.join('results', 'union_mode')
    output_path = os.path.join('results')
    for only_best in [True, False]:
        results = load_just_one_result(input_path, dataset='synthetic_data', metric='sil')
        save_just_one_result(results, output_path, dataset='synthetic_data', metric='sil')

main()