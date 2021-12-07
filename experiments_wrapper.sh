#!/bin/bash

python scenarios_generator.py
python experiments_launcher.py
python results_processors/optimization_results_summarizer.py
python results_processors/diversificator.py