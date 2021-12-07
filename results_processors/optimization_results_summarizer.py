from __future__ import print_function

import itertools
import os 
import sys

import pandas as pd
import yaml

from six import iteritems
from results_processors_utils import load_result
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..' )
sys.path.append( mymodule_dir )
from utils import get_scenario_info, create_directory, SCENARIO_PATH, OPTIMIZATION_RESULT_PATH

def main():
    
    scenarios, _, _ = get_scenario_info()

    scenario_with_results = {k: v for k, v in iteritems(scenarios) if v['status'] == 'Ok' and v['results'] is not None}

    results = {}
    for scenario in scenario_with_results.keys():
        with open(os.path.join(SCENARIO_PATH, scenario), 'r') as stream:
            try:
                c = yaml.safe_load(stream)
                optimization_approach = 'exhaustive' if c['optimization']['budget'] == 'inf' else 'smbo'
                dataset = c['general']['dataset']
                optimization_internal_metric = c['optimization']['metric']
                if optimization_approach not in results:
                    results[optimization_approach] = {dataset: [optimization_internal_metric]}
                else:
                    results[optimization_approach][dataset].append(optimization_internal_metric)
            except yaml.YAMLError as exc:
                print(exc)

    for optimization_approach in results.keys():
        input_path = os.path.join(OPTIMIZATION_RESULT_PATH, optimization_approach)
        output_path = create_directory(input_path, 'summary')
        output_file_name = 'summary.csv'
        summary = pd.DataFrame()
        for dataset, optimization_internal_metrics in results[optimization_approach].items():
            for optimization_internal_metric in optimization_internal_metrics:
                try:
                    summary = summary.append(load_result(input_path, dataset=dataset, metric=optimization_internal_metric))
                except Exception as e:
                    print(e)
        summary.to_csv(os.path.join(output_path, output_file_name), index=False)

main()