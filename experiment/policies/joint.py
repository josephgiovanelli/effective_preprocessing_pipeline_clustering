from experiment.policies.policy import Policy
from experiment.algorithm import space as ALGORITHM_SPACE
from experiment.objective import objective_joint

import functools

from hyperopt import tpe, fmin, Trials
import hyperopt.pyll.stochastic

import numpy as np

class Joint(Policy):

    def run(self, X, y):
        super(Joint, self).run(X, y)
        current_pipeline_configuration = {}
        current_algo_configuration = {}
        trials = Trials()
        algorithm = self.config['algorithm']
        space = {
            'pipeline': self.PIPELINE_SPACE,
            'algorithm': ALGORITHM_SPACE.get_domain_space(algorithm, self.max_k),
        }
        obj_pl = functools.partial(objective_joint,
                algorithm=self.config['algorithm'],
                X=X,
                y=y,
                context=self.context,
                config=self.config)
        fmin(
            fn=obj_pl, 
            space=space, 
            algo=tpe.suggest, 
            max_evals=None if self.config['budget'] == 'time' else self.config['runtime'],
            max_time=self.config['runtime'] if self.config['budget'] == 'time' else None,     
            trials=trials,
            show_progressbar=False,
            verbose=0,
            rstate=np.random.RandomState(self.config['seed'])
        )

        best_config = self.context['best_config']
        super(Joint, self).display_step_results(best_config)
        current_pipeline_configuration = best_config['pipeline']