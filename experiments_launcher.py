import psutil
import json
import os
import shutil
import yaml
import subprocess
import datetime
import argparse
import time
import experiment

from six import iteritems
from prettytable import PrettyTable
from tqdm import tqdm
from results_processors.utils import create_directory

from experiment.utils import scenarios as scenarios_util


RESULT_PATH = create_directory('results', 'optimization')
SCENARIO_PATH = './scenarios'
GLOBAL_SEED = 42

# Gather list of scenarios
scenario_list = [p for p in os.listdir(SCENARIO_PATH) if '.yaml' in p]
scenarios = {}

# Determine which one have no result files
for scenario in scenario_list:
    base_scenario = scenario.split('.yaml')[0]
    scenarios[scenario] = {'results': None, 'path': scenario}
    config = experiment.utils.scenarios.to_config(
        experiment.utils.scenarios.load(os.path.join(SCENARIO_PATH, scenario)))
    result_path = create_directory(RESULT_PATH, 'exhaustive' if config['runtime'] == 'inf' else 'smbo')
    unit_measure = config['budget']
    result_list = [p for p in os.listdir(result_path) if '.json' in p]
    for result in result_list:
        base_result = result.split('.json')[0]
        if base_result.__eq__(base_scenario):
            scenarios[scenario]['results'] = result
            #date = base_result.split(base_scenario + '_')[-1].replace('_', ' ')
            #scenarios[scenario]['results_date'] = date

# Calculate total amount of time
total_runtime = 0
for path, scenario in iteritems(scenarios):
    with open(os.path.join(SCENARIO_PATH, path), 'r') as f:
        details = None
        try:
            details = yaml.safe_load(f)
        except Exception:
            details = None
            scenario['status'] = 'Invalid YAML'
        if details is not None:
            try:
                runtime = 1 if details['setup']['runtime'] == 'inf' else details['setup']['runtime']
                scenario['status'] = 'Ok'
                scenario['runtime'] = runtime
                if scenario['results'] is None:
                    total_runtime += runtime
            except:
                scenario['status'] = 'No runtime info'

# Display list of scenario to be run
invalid_scenarios = {k: v for k, v in iteritems(
    scenarios) if v['status'] != 'Ok'}
t_invalid = PrettyTable(['PATH', 'STATUS'])
t_invalid.align["PATH"] = "l"
for v in invalid_scenarios.values():
    t_invalid.add_row([v['path'], v['status']])

scenario_with_results = {k: v for k, v in iteritems(
    scenarios) if v['status'] == 'Ok' and v['results'] is not None}
t_with_results = PrettyTable(['PATH', 'RUNTIME',  'STATUS', 'RESULTS'])
t_with_results.align["PATH"] = "l"
t_with_results.align["RESULTS"] = "l"
for v in scenario_with_results.values():
    t_with_results.add_row(
        [v['path'], str(v['runtime']), v['status'], v['results']])

to_run = {k: v for k, v in iteritems(
    scenarios) if v['status'] == 'Ok' and v['results'] is None}
t_to_run = PrettyTable(['PATH', 'RUNTIME', 'STATUS'])
t_to_run.align["PATH"] = "l"
for v in to_run.values():
    t_to_run.add_row([v['path'], str(v['runtime']), v['status']])

print('# INVALID SCENARIOS')
print(t_invalid)

print
print('# SCENARIOS WITH AVAILABLE RESULTS')
print(t_with_results)

print
print('# SCENARIOS TO BE RUN')
print(t_to_run)
print(f'TOTAL RUNTIME: {total_runtime} {unit_measure}')
print

with tqdm(total=total_runtime) as pbar:
    for info in to_run.values():
        base_scenario = info['path'].split('.yaml')[0]
        output = base_scenario.split('_')[0]
        pbar.set_description("Running scenario {}\n\r".format(info['path']))
        print()

        current_scenario = scenarios_util.load(
            os.path.join(SCENARIO_PATH, info['path']))
        config = scenarios_util.to_config(current_scenario)

        result_path = os.path.join(RESULT_PATH, 'exhaustive' if config['runtime'] == 'inf' else 'smbo')

        cmd = 'python ./main.py -s {} -c control.seed={} -p {} -r {}'.format(
            os.path.join(SCENARIO_PATH, info['path']),
            GLOBAL_SEED,
            "features normalize outlier",
            result_path)
        with open(os.path.join(result_path, '{}_stdout.txt'.format(base_scenario)), "a") as log_out:
            with open(os.path.join(result_path, '{}_stderr.txt'.format(base_scenario)), "a") as log_err:
                start_time = time.time()
                process = subprocess.call(
                    cmd, shell=True, stdout=log_out, stderr=log_err)
                print("--- %s seconds ---" % (time.time() - start_time))
        
        pbar.update(info['runtime'])
