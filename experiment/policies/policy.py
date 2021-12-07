from experiment.pipeline.PrototypeSingleton import PrototypeSingleton

import json

class Policy(object):
    def __init__(self, config):
        self.PIPELINE_SPACE = PrototypeSingleton.getInstance().getDomainSpace()
        self.max_k = PrototypeSingleton.getInstance().getX().shape[0] / 2
        self.compute_baseline = False
        self.config = config
        self.context = {
            'iteration': 0,
            'history_hash': [],
            'history_index': {},
            'history': [],
            'max_history_internal_metric': float('-inf'),
            'max_history_external_metric': float('-inf'),
            'best_config': {},
        }

    def __compute_baseline(self, X, y):
        raise Exception('No implementation for baseline score')

    def run(self, X, y):
        if self.compute_baseline:
            self.__compute_baseline(X, y)

    def display_step_results(self, best_config):
        print('#' * 50)
        print('BEST PIPELINE:\n {}'.format(json.dumps(best_config['pipeline'], indent=4, sort_keys=True),))
        print('BEST ALGORITHM:\n {}'.format(json.dumps(best_config['algorithm'], indent=4, sort_keys=True)))
        print('BEST INTERNAL: {}, EXTERNAL: '.format(round(best_config['internal_metric'], 4), round(best_config['external_metric'], 4)))
        print('#' * 50)
