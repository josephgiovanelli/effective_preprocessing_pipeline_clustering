from hyperopt import hp

def generate_domain_space(prototype):
    domain_space = {}
    for k, v in prototype.items():
        domain_space[k] = hp.choice(k, v)
    return domain_space

def generate_union_domain_space(prototype):
    algorithms_space = []
    for algorithm in prototype.keys():
        algorithm_config = {}
        for k, v in prototype[algorithm].items():
            algorithm_config[k] = hp.choice('{}_{}'.format(algorithm, k), v)
        algorithms_space.append((algorithm, algorithm_config))
    return hp.choice('algorithm', algorithms_space)


