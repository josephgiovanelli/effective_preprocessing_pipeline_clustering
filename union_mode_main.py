from sklearn.impute import SimpleImputer

from experiment.pipeline.PrototypeSingleton import PrototypeSingleton
from experiment.utils import scenarios, serializer, cli, datasets
from experiment import policies

import json

import openml

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os


def load_dataset(id, kind):
    if kind == 'openml':
        dataset = openml.datasets.get_dataset(id)
        X, y, categorical_indicator, _ = dataset.get_data(
            dataset_format='array',
            target=dataset.default_target_attribute
        )
        print(dataset.name)
    else:    
        X, y = datasets.get_dataset(id)
        categorical_indicator = [False for _ in range(X.shape[1])]
        print(id)
    print(X)
    print(y)
    num_features = [i for i, x in enumerate(categorical_indicator) if x == False]
    cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
    print("numeriche: " + str(len(num_features)) + " categoriche: " + str(len(cat_features)))
    PrototypeSingleton.getInstance().setFeatures(num_features, cat_features)
    PrototypeSingleton.getInstance().set_X_y(X, y)
    return X, y

def main(args):
    scenario = scenarios.load(args.scenario)
    scenario = cli.apply_scenario_customization(scenario, args.customize)
    config = scenarios.to_config(scenario)

    print("config time: " + str(config['runtime']))
    print('SCENARIO:\n {}'.format(json.dumps(scenario, indent=4, sort_keys=True)))

    PrototypeSingleton.getInstance().setPipeline(args.pipeline)

    X, y = load_dataset(config['dataset'], config['dataset_kind'])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=scenario['control']['seed']
    )

    policy = policies.initiate(config['policy'], config)
    policy.run(X, y)

    serializer.serialize_results(scenario, policy, args.result_path, args.pipeline)

if __name__ == "__main__":
    args = cli.parse_args()
    main(args)