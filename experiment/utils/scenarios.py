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
	config = {
		'dataset_kind': scenario['general']['dataset_kind'],
		'dataset': scenario['general']['dataset'],
		'seed': scenario['general']['seed'],
		'metric': scenario['optimization']['metric'],
		'budget_kind': scenario['optimization']['budget_kind'],
		'budget': scenario['optimization']['budget'],
	}
	return config