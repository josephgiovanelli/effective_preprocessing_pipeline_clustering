import functools
import hyperopt.pyll.stochastic
import numpy as np

from experiment.policies.policy import Policy
from experiment.algorithm import space as ALGORITHM_SPACE
from experiment.objective import objective_union
from functools import partial
from hyperopt import tpe, fmin, Trials

from experiment.utils.exhaustive_search import suggest


class Union(Policy):

    def run(self, X, y):
        super(Union, self).run(X, y)
        current_pipeline_configuration = {}
        current_algo_configuration = {}
        trials = Trials()
        space = {
            'pipeline': self.PIPELINE_SPACE,
            'algorithm': ALGORITHM_SPACE.get_domain_space("union", self.max_k),
        }
        obj_pl = functools.partial(objective_union,
                X=X,
                y=y,
                context=self.context,
                config=self.config)

        fmin(
            fn=obj_pl, 
            space=space, 
            algo=partial(suggest, nbMaxSucessiveFailures=1000) if self.config['runtime'] == 'inf' else tpe.suggest, 
            max_evals=np.inf if self.config['runtime'] == 'inf' else (None if self.config['budget'] == 'time' else self.config['runtime']),
            max_time=None if self.config['runtime'] == 'inf' else (self.config['runtime'] if self.config['budget'] == 'time' else None),     
            trials=trials,
            show_progressbar=False,
            verbose=0,
            rstate=np.random.RandomState(self.config['seed'])
        )

        best_config = self.context['best_config']
        super(Union, self).display_step_results(best_config)
        current_pipeline_configuration = best_config['pipeline']