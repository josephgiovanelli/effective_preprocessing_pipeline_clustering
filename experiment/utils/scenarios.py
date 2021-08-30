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
			'time': scenario['setup']['runtime'],
			'algorithm': scenario['setup']['algorithm'],
		}
	except:
		config = {
			'seed': scenario['control']['seed'],
			'time': scenario['setup']['runtime'],
		}

	if scenario['policy'] is not None:
		config.update(scenario['policy'])
	return config