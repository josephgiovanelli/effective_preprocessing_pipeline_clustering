from __future__ import print_function

import itertools
import os 

from results_extraction_utils import load_results, save_results
from utils import create_directory


def main():
    input_path = os.path.join('results', 'union_mode')
    output_path = os.path.join('results')
    for only_best in [True, False]:
        results = load_results(input_path, only_best)
        save_results(results, output_path, only_best)

main()