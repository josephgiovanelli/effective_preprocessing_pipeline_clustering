from __future__ import print_function
import datetime

import itertools
import os 
import sys
import time

import pandas as pd
import yaml

from six import iteritems
from results_processors_utils import load_result
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..' )
sys.path.append( mymodule_dir )
from utils import get_scenario_info, create_directory, SCENARIO_PATH, OPTIMIZATION_RESULT_PATH

def main():
    
    scenarios = get_scenario_info()

    scenario_with_results = {k: v for k, v in iteritems(scenarios) if v['results'] is not None}

    results = {}
    for _, scenario in scenario_with_results.items():
        with open(os.path.join(SCENARIO_PATH, scenario['path']), 'r') as stream:
            try:
                c = yaml.safe_load(stream)
                dataset = c['general']['dataset']
                for optimization_method, optimization_conf in c['optimizations'].items():
                    optimization_internal_metric = optimization_conf['metric']
                    results[optimization_method] = {dataset: [optimization_internal_metric]}
            except yaml.YAMLError as exc:
                print(exc)

    for optimization_method in results.keys():
        input_path = os.path.join(OPTIMIZATION_RESULT_PATH, optimization_method)
        output_path = create_directory(input_path, 'summary')
        output_file_name = 'summary.csv'
        print(f'Summarizing {optimization_method} runs')
        try:
            summary = pd.read_csv(os.path.join(output_path, output_file_name))
            print('\tA previous summarization was found')
        except:
            print('\tA previous summarization was not found')
            print('\tSummarization process starts')
            summary = pd.DataFrame()
            start_time = time.time()
            for dataset, optimization_internal_metrics in results[optimization_method].items():
                print(f'\t\tdataset: {dataset}\n\t\t\tinternal_metrics: {optimization_internal_metrics}')
                for optimization_internal_metric in optimization_internal_metrics:
                    try:
                        summary = summary.append(load_result(input_path, dataset=dataset, metric=optimization_internal_metric))
                    except Exception as e:
                        print(e)
            end_time = time.time()
            duration = int(end_time) - int(start_time)
            print(f'Summarization process ends: {datetime.timedelta(seconds=duration)}\n')
            summary.to_csv(os.path.join(output_path, output_file_name), index=False)

main()