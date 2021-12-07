import os
import subprocess
import time

from six import iteritems
from prettytable import PrettyTable
from tqdm import tqdm

from utils import get_scenario_info, SCENARIO_PATH, OPTIMIZATION_RESULT_PATH
from experiment.utils import scenarios as scenarios_util

scenarios, total_budget, unit_measure = get_scenario_info()

# Display list of scenario to be run
invalid_scenarios = {k: v for k, v in iteritems(
    scenarios) if v['status'] != 'Ok'}
t_invalid = PrettyTable(['PATH', 'STATUS'])
t_invalid.align["PATH"] = "l"
for v in invalid_scenarios.values():
    t_invalid.add_row([v['path'], v['status']])

scenario_with_results = {k: v for k, v in iteritems(
    scenarios) if v['status'] == 'Ok' and v['results'] is not None}
t_with_results = PrettyTable(['PATH', 'BUDGET',  'STATUS', 'RESULTS'])
t_with_results.align["PATH"] = "l"
t_with_results.align["RESULTS"] = "l"
for v in scenario_with_results.values():
    t_with_results.add_row(
        [v['path'], str(v['budget']), v['status'], v['results']])

to_run = {k: v for k, v in iteritems(
    scenarios) if v['status'] == 'Ok' and v['results'] is None}
t_to_run = PrettyTable(['PATH', 'BUDGET', 'STATUS'])
t_to_run.align["PATH"] = "l"
for v in to_run.values():
    t_to_run.add_row([v['path'], str(v['budget']), v['status']])

print('# INVALID SCENARIOS')
print(t_invalid)

print
print('# SCENARIOS WITH AVAILABLE RESULTS')
print(t_with_results)

print
print('# SCENARIOS TO BE RUN')
print(t_to_run)
print(f'TOTAL budget: {total_budget} {unit_measure}')
print

with tqdm(total=total_budget) as pbar:
    for info in to_run.values():
        base_scenario = info['path'].split('.yaml')[0]
        output = base_scenario.split('_')[0]
        pbar.set_description("Running scenario {}\n\r".format(info['path']))
        print()

        current_scenario = scenarios_util.load(os.path.join(SCENARIO_PATH, info['path']))
        config = scenarios_util.to_config(current_scenario)

        result_path = os.path.join(OPTIMIZATION_RESULT_PATH, 'exhaustive' if config['budget'] == 'inf' else 'smbo')

        cmd = 'python ./main.py -s {} -p {} -r {}'.format(
            os.path.join(SCENARIO_PATH, info['path']),
            "features normalize outlier",
            result_path)
        with open(os.path.join(result_path, '{}_stdout.txt'.format(base_scenario)), "a") as log_out:
            with open(os.path.join(result_path, '{}_stderr.txt'.format(base_scenario)), "a") as log_err:
                start_time = time.time()
                process = subprocess.call(
                    cmd, shell=True, stdout=log_out, stderr=log_err)
                print("--- %s seconds ---" % (time.time() - start_time))
        
        pbar.update(info['budget'])
