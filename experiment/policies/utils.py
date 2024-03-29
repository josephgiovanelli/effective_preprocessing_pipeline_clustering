from experiment.policies.iterative import Iterative
from experiment.policies.split import Split
from experiment.policies.adaptive import Adaptive
from experiment.policies.joint import Joint
from experiment.policies.union import Union

def initiate(name, config):
    policies = {
        'iterative': Iterative,
        'split': Split,
        'adaptive': Adaptive,
        'joint': Joint,
        'union': Union
    }
    if name in policies:
        return policies[name](config)
    else:
        print('Invalid dataset. Possible choices: {}'.format(
            ', '.join(policies.keys())
        ))
        exit(1)  # TODO: Throw exception