import json
import os
import shutil
import yaml
import subprocess
import datetime
import argparse

from six import iteritems
from prettytable import PrettyTable
from tqdm import tqdm
from results_processors.utils import create_directory
from auto_pipeline_builder import pseudo_exhaustive_pipelines

from experiment.utils import scenarios as scenarios_util


parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")

RESULT_PATH = "./results/"
RESULT_PATH = create_directory(RESULT_PATH, "union_mode")
SCENARIO_PATH = "./scenarios/union_mode"
GLOBAL_SEED = 42

def yes_or_no(question):
    while True:
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

# Gather list of scenarios
scenario_list = [p for p in os.listdir(SCENARIO_PATH) if '.yaml' in p]
result_list = [p for p in os.listdir(RESULT_PATH) if '.json' in p]
scenarios = {}

# Determine which one have no result files
for scenario in scenario_list:
    base_scenario = scenario.split('.yaml')[0]
    if scenario not in scenarios:
        scenarios[scenario] = {'results': None, 'path': scenario}
    for result in result_list:
        base_result = result.split('.json')[0]
        print(base_scenario)
        print(base_result)
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
                runtime = details['setup']['runtime'] * 5
                scenario['status'] = 'Ok'
                scenario['runtime'] = runtime
                if scenario['results'] is None:
                    total_runtime += runtime
            except:
                scenario['status'] = 'No runtime info'

# Display list of scenario to be run
invalid_scenarios = {k:v for k,v in iteritems(scenarios) if v['status'] != 'Ok'}
t_invalid = PrettyTable(['PATH', 'STATUS'])
t_invalid.align["PATH"] = "l"
for v in invalid_scenarios.values():
    t_invalid.add_row([v['path'], v['status']])

scenario_with_results = {k:v for k,v in iteritems(scenarios) if v['status'] == 'Ok' and v['results'] is not None}
t_with_results = PrettyTable(['PATH', 'RUNTIME',  'STATUS', 'RESULTS'])
t_with_results.align["PATH"] = "l"
t_with_results.align["RESULTS"] = "l"
for v in scenario_with_results.values():
    t_with_results.add_row([v['path'], str(v['runtime']) + 's', v['status'], v['results']])

to_run = {k:v for k,v in iteritems(scenarios) if v['status'] == 'Ok' and v['results'] is None}
t_to_run = PrettyTable(['PATH', 'RUNTIME', 'STATUS'])
t_to_run.align["PATH"] = "l"
for v in to_run.values():
    t_to_run.add_row([v['path'], str(v['runtime']) + 's', v['status']])

print('# INVALID SCENARIOS')
print(t_invalid)

print
print('# SCENARIOS WITH AVAILABLE RESULTS')
print(t_with_results)

print
print('# SCENARIOS TO BE RUN')
print(t_to_run)
print('TOTAL RUNTIME: {} ({}s)'.format(datetime.timedelta(seconds=total_runtime), total_runtime))
print

print("The total runtime is {}.".format(datetime.timedelta(seconds=total_runtime)))
print

import psutil


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

with tqdm(total=total_runtime) as pbar:
    for info in to_run.values():
        base_scenario = info['path'].split('.yaml')[0]
        output = base_scenario.split('_')[0]
        pbar.set_description("Running scenario {}\n\r".format(info['path']))
        print()

        current_scenario = scenarios_util.load(os.path.join(SCENARIO_PATH, info['path']))
        config = scenarios_util.to_config(current_scenario)

        pipelines = pseudo_exhaustive_pipelines()
        results = []

        for i in range(0, len(pipelines)):
            pipeline = pipelines[i]
            cmd = 'python ./union_mode_main.py -s {} -c control.seed={} -p {} -r {}'.format(
                os.path.join(SCENARIO_PATH, info['path']),
                GLOBAL_SEED,
                pipeline,
                "./results/")
            with open(os.path.join(RESULT_PATH, '{}_stdout.txt'.format(base_scenario + "_" + str(i))),
                        "a") as log_out:
                with open(os.path.join(RESULT_PATH, '{}_stderr.txt'.format(base_scenario + "_" + str(i))),
                            "a") as log_err:
                    try:
                        process = subprocess.Popen(cmd, shell=True, stdout=log_out, stderr=log_err)
                    except:
                        kill(process.pid)
                        print("\n\n" + base_scenario + " didn't finished\n\n")

            try:
                os.rename(os.path.join(RESULT_PATH, '{}.json'.format(base_scenario)),
                            os.path.join(RESULT_PATH, '{}.json'.format(base_scenario + "_" + str(i))))

                with open(
                        os.path.join(RESULT_PATH, '{}.json'.format(base_scenario + "_" + str(i)))) as json_file:
                    data = json.load(json_file)
                    accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                    results.append(accuracy)
            except:
                accuracy = 0
                results.append(accuracy)
            print(results)

        try:
            max_i = 0
            for i in range(1, len(pipelines)):
                if results[i] > results[max_i]:
                    max_i = i

            src_dir = os.path.join(RESULT_PATH, '{}.json'.format(base_scenario + "_" + str(max_i)))
            dst_dir = os.path.join(RESULT_PATH, '{}.json'.format(base_scenario + "_best_pipeline_" + str(max_i)))
            shutil.copy(src_dir, dst_dir)
        except:
            with open(os.path.join(RESULT_PATH, '{}.txt'.format(base_scenario + "_best_pipeline")), "a") as log_out:
                log_out.write("trying to get the best pipeline: no available result")

        pbar.update(info['runtime'])
