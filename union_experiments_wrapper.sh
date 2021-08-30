#!/bin/bash

python union_scenarios_generator.py
python union_experiments_launcher.py
cd results_processors
cd union_mode
python results_comparator.py