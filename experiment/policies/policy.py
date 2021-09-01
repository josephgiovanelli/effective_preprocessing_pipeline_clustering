from experiment.objective import get_baseline_score
from experiment.pipeline.PrototypeSingleton import PrototypeSingleton

import json

class Policy(object):
    def __init__(self, config):
        self.PIPELINE_SPACE = PrototypeSingleton.getInstance().getDomainSpace()
        if config['runtime'] == 400 and config['budget'] == 'time' and config['policy'] == 'split':
            self.compute_baseline = True
        else:
            self.compute_baseline = False
        self.config = config
        self.context = {
            'iteration': 0,
            'history_hash': [],
            'history_index': {},
            'history': [],
            'max_history_score': 0.,
            'max_history_step': 'baseline',
            'max_history_score_ami': 0.,
            'best_config': {},
        }

    def __compute_baseline(self, X, y):
        baseline_score, baseline_score_std = get_baseline_score(
            self.config['algorithm'],
            X,
            y,
            self.config['seed'])
        self.context['baseline_score'] = baseline_score

    def run(self, X, y):
        if self.compute_baseline:
            self.__compute_baseline(X, y)

    def display_step_results(self, best_config):
        print('{} STEP RESULT {}'.format('#' * 20, '#' * 20))
        print('BEST PIPELINE:\n {}'.format(json.dumps(best_config['pipeline'], indent=4, sort_keys=True),))
        print('BEST ALGO CONFIG:\n {}'.format(json.dumps(best_config['algorithm'], indent=4, sort_keys=True)))
        print('BEST SCORE: {}'.format(best_config['score']))
        print('#' * 50)
