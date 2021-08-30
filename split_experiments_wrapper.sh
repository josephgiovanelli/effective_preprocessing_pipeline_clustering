#!/bin/bash

python split_scenarios_generator.py -mode algorithm
python split_scenarios_generator.py -mode preprocessing_algorithm
python split_experiments_launcher.py -mode algorithm
python split_experiments_launcher.py -mode preprocessing_algorithm
cd results_processors
cd split_mode
python results_comparator.py