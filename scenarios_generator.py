import os
import copy
import re
import argparse

import pandas as pd

from collections import OrderedDict
from results_processors.utils import create_directory


parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")

SCENARIO_PATH = './scenarios/'

policies = ['union']
metrics = ['SIL', 'CH', 'DBI']
datasets = ['wine', 'seeds', 'parkinsons', 'iris', 'breast', 'synthetic_data']

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
    ('title', ''),
    ('setup', {
        'policy': 'union',
        'runtime': 'inf',
        'budget': 'iterations',
        'dataset': '',
        'dataset_kind': 'uci',
        'metric': ''
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

for dataset in datasets:
    print('# DATASET: {}'.format(dataset))
    for metric in metrics:
        scenario = copy.deepcopy(base)
        scenario['setup']['dataset'] = dataset
        scenario['setup']['metric'] = metric
        scenario['title'] = 'AutoML on {} with {} policy'.format(
            dataset,
            scenario['setup']['policy'].title()
        )

        path = os.path.join(SCENARIO_PATH, '{}_{}.yaml'.format(dataset, metric.lower()))
        __write_scenario(path, scenario)

