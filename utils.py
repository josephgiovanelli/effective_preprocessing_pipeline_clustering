import os
import yaml
import experiment

from six import iteritems

from experiment.utils import scenarios as scenarios_util

RESULT_PATH = './results'
SCENARIO_PATH = './scenarios'

def create_directory(result_path, directory):
    result_path = os.path.join(result_path, directory)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path

OPTIMIZATION_RESULT_PATH = create_directory(RESULT_PATH, 'optimization')
DIVERSIFICATION_RESULT_PATH = create_directory(RESULT_PATH, 'diversification')

def get_scenario_info():
    # Gather list of scenarios
    scenario_list = [p for p in os.listdir(SCENARIO_PATH) if '.yaml' in p]
    scenarios = {}

    # Determine which one have no result files
    for scenario in scenario_list:
        base_scenario = scenario.split('.yaml')[0]
        scenarios[scenario] = {'results': None, 'path': scenario}
        config = experiment.utils.scenarios.to_config(
            experiment.utils.scenarios.load(os.path.join(SCENARIO_PATH, scenario)))
        relative_result_path = create_directory(OPTIMIZATION_RESULT_PATH, 'exhaustive' if config['budget'] == 'inf' else 'smbo')
        unit_measure = config['budget_kind']
        result_list = [p for p in os.listdir(relative_result_path) if '.json' in p]
        for result in result_list:
            base_result = result.split('.json')[0]
            if base_result.__eq__(base_scenario):
                scenarios[scenario]['results'] = result

    # Calculate total amount of time
    total_budget = 0
    for path, scenario in iteritems(scenarios):
        with open(os.path.join(SCENARIO_PATH, path), 'r') as f:
            details = None
            try:
                details = yaml.safe_load(f)
            except Exception:
                details = None
                scenario['status'] = 'Invalid YAML'
            if details is not None:
                try:
                    budget = 1 if details['optimization']['budget'] == 'inf' else details['optimization']['budget']
                    scenario['status'] = 'Ok'
                    scenario['budget'] = budget
                    if scenario['results'] is None:
                        total_budget += budget
                except:
                    scenario['status'] = 'No budget info'
    return scenarios, total_budget, unit_measure