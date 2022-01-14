import json
import openml
import os

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from experiment.pipeline.PrototypeSingleton import PrototypeSingleton
from experiment.utils import scenarios, serializer, cli, datasets
from experiment import policies
from sklearn.model_selection import train_test_split


def main(args):
    scenario = scenarios.load(args.scenario)
    config = scenarios.to_config(scenario, args.optimization_approach)
    print('SCENARIO:')
    print('#' * 50)
    print(f'{json.dumps(scenario, indent=4, sort_keys=True)}')
    print('#' * 50 + '\n')

    X, y = datasets.get_dataset(config['dataset'])
    PrototypeSingleton.getInstance().setPipeline(args.pipeline)

    config['result_path'] = args.result_path
    policy = policies.initiate(config)
    policy.run(X, y)

    serializer.serialize_results(scenario, policy, os.path.join(
        args.result_path, config['dataset'] + '_' + config['metric'] + '.json'), args.pipeline)


if __name__ == "__main__":
    args = cli.parse_args()
    main(args)
