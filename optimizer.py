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


def load_dataset(id, kind):
    if kind == 'openml':
        dataset = openml.datasets.get_dataset(id)
        X, y, categorical_indicator, _ = dataset.get_data(
            dataset_format='array',
            target=dataset.default_target_attribute
        )
        dataset_name = dataset.name
        dataset_features_names = [str(elem)
                                  for elem in list(dataset.features.values())]
        dataset_features_names = dataset_features_names[:-1]
    elif  kind == 'uci':
        X, y, dataset_features_names = datasets.get_dataset(id)
        categorical_indicator = [False for _ in range(X.shape[1])]
        dataset_name = id
    else:
        raise Exception(f'''missing dataset kind for dataset {id}''')
    num_features = [i for i, x in enumerate(categorical_indicator) if x == False]
    cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
    PrototypeSingleton.getInstance().setFeatures(num_features, cat_features)
    PrototypeSingleton.getInstance().set_X_y(X, y)
    PrototypeSingleton.getInstance().setDatasetFeaturesName(dataset_features_names)
    print('DATASET:')
    print('#' * 50)
    print(f'\tname:\t{dataset_name}')
    print(f'\tshape:\t{X.shape}')
    print(f'\tnumerical features:\t{len(num_features)}\n\tcategorical features:\t{len(cat_features)}')
    print(f'\tfirst instance of X:\t{X[0, :]}')
    print(f'\tfirst instance of y:\t{y[0]}')
    print('#' * 50 + '\n')
    return X, y


def main(args):
    scenario = scenarios.load(args.scenario)
    config = scenarios.to_config(scenario, args.optimization_approach)
    print('SCENARIO:')
    print('#' * 50)
    print(f'{json.dumps(scenario, indent=4, sort_keys=True)}')
    print('#' * 50 + '\n')

    X, y = load_dataset(config['dataset'], config['dataset_kind'])
    PrototypeSingleton.getInstance().setPipeline(args.pipeline)

    config['result_path'] = args.result_path
    policy = policies.initiate(config)
    policy.run(X, y)

    serializer.serialize_results(scenario, policy, os.path.join(
        args.result_path, config['dataset'] + '_' + config['metric'] + '.json'), args.pipeline)


if __name__ == "__main__":
    args = cli.parse_args()
    main(args)
