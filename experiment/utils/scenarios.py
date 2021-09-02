import yaml

def load(path):
	scenario = None
	with open(path, 'r') as f:
	    try:
	        scenario = yaml.safe_load(f)
	    except yaml.YAMLError as exc:
	        print(exc)
	        scenario = None
	if scenario is not None:
		scenario['file_name'] = path.split('/')[-1].split('.')[0]
	return scenario

def validate(scenario):
	return True #  TODO

def to_config(scenario):
	try:
		config = {
			'seed': scenario['control']['seed'],
			'runtime': scenario['setup']['runtime'],
			'budget': scenario['setup']['budget'],
			'algorithm': scenario['setup']['algorithm'],
			'policy': scenario['setup']['policy'],
			'dataset': scenario['setup']['dataset'],
			'dataset_kind': scenario['setup']['dataset_kind'],
			'metric': scenario['setup']['metric'],
		}
	except:
		config = {
			'seed': scenario['control']['seed'],
			'runtime': scenario['setup']['runtime'],
			'budget': scenario['setup']['budget'],
			'policy': scenario['setup']['policy'],
			'dataset': scenario['setup']['dataset'],
			'dataset_kind': scenario['setup']['dataset_kind'],
			'metric': scenario['setup']['metric'],
		}

	if scenario['policy'] is not None:
		config.update(scenario['policy'])
	return config