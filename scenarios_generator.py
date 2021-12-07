import os
import copy

from collections import OrderedDict
from utils import SCENARIO_PATH

#metrics = ['sil', 'ch', 'dbi']
#datasets = ['wine', 'seeds', 'parkinsons', 'iris', 'breast', 'synthetic_data']
datasets = ['iris']
optimization_metrics = ['sil']

base = OrderedDict([
    ('general', {
        'dataset_kind': 'uci',
        'dataset': '',
        'seed': 42,
    }),
    ('optimization', {
        'metric': '',
        'budget_kind': 'iterations',
        'budget': 335,
        #'budget': 'inf',
    }),
    ('diversification', {
        'num_results': 3,
        'method': 'mmr',
        'lambda': 0.5,
        'criterion': 'features_set_n_clusters',
        'metric': 'euclidean',
    })
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
    for metric in optimization_metrics:
        scenario = copy.deepcopy(base)
        scenario['general']['dataset'] = dataset
        scenario['optimization']['metric'] = metric

        path = os.path.join(SCENARIO_PATH, '{}_{}.yaml'.format(dataset, metric))
        __write_scenario(path, scenario)

