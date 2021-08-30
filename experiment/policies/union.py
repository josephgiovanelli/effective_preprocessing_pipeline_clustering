from experiment.policies.policy import Policy
from experiment.algorithm import space as ALGORITHM_SPACE
from experiment.objective import objective_union

import functools

from hyperopt import tpe, fmin, Trials
import hyperopt.pyll.stochastic

class Union(Policy):

    def run(self, X, y):
        super(Union, self).run(X, y)
        current_pipeline_configuration = {}
        current_algo_configuration = {}
        trials = Trials()
        space = {
            'pipeline': self.PIPELINE_SPACE,
            'algorithm': ALGORITHM_SPACE.get_domain_space("union"),
        }
        obj_pl = functools.partial(objective_union,
                X=X,
                y=y,
                context=self.context,
                config=self.config)
        fmin(
            fn=obj_pl, 
            space=space, 
            algo=tpe.suggest, 
            max_evals=None,
            max_time=self.config['time'],     
            trials=trials,
            show_progressbar=False,
            verbose=0
        )

        best_config = self.context['best_config']
        super(Union, self).display_step_results(best_config)
        current_pipeline_configuration = best_config['pipeline']