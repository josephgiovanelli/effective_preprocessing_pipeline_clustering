import os
import copy
import re
import argparse

import pandas as pd

from commons import large_comparison_classification_tasks, extended_benchmark_suite, benchmark_suite, algorithms
from collections import OrderedDict
from results_processors.utils import create_directory


parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")


SCENARIO_PATH = './scenarios/'
SCENARIO_PATH = create_directory(SCENARIO_PATH, "union_mode")

policies = ['union']


policies_config = {
    'iterative': {
        'step_algorithm': 15,
        'step_pipeline': 15,
        'reset_trial': False
    },
    'split': {
        'step_pipeline': 30
    },
    'adaptive': {
        'initial_step_time': 15,
        'reset_trial': False,
        'reset_trials_after': 2
    },
    'joint': {},
    'union': {}
}

base = OrderedDict([
    ('title', 'Random Forest on Wine with Iterative policy'),
    ('setup', {
        'policy': 'iterative',
        'runtime': 400,
        'dataset': 'wine'
    }),
    ('control', {
        'seed': 42
    }),
    ('policy', {})
])

def __write_scenario(path, scenario):
    try:
        print('   -> {}'.format(path))
        with open(path, 'w') as f:
            for k,v in scenario.items():
                if isinstance(v, str):
                    f.write('{}: {}\n'.format(k, v))
                else:
                    f.write('{}:\n'.format(k))
                    for i,j in v.items():
                        f.write('  {}: {}\n'.format(i,j))
    except Exception as e:
        print(e)

def get_filtered_datasets():
    df = pd.read_csv("results_processors/meta_features/simple-meta-features.csv")
    df = df.loc[df['did'].isin(list(dict.fromkeys(extended_benchmark_suite + [10, 20, 26] + [15, 29, 1053, 1590])))]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()

for id in get_filtered_datasets():
    print('# DATASET: {}'.format(id))
    for policy in policies:
        scenario = copy.deepcopy(base)
        scenario['setup']['dataset'] = id
        scenario['setup']['policy'] = policy
        scenario['policy'] = copy.deepcopy(policies_config[policy])
        scenario['title'] = 'AutoML on dataset n {} with {} policy'.format(
            id,
            policy.title()
        )
        runtime = scenario['setup']['runtime']

        path = os.path.join(SCENARIO_PATH, '{}.yaml'.format(id))
        __write_scenario(path, scenario)

