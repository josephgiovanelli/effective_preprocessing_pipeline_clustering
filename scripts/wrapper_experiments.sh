#!/bin/bash

# printf '\n\nOPTIMIZATION\n\n'
# python optimization_launcher.py

printf '\n\nSUMMARIZATION\n\n'
python results_processors/optimization_results_summarizer.py

printf '\n\nDIVERSIFICATION\n\n'
python results_processors/diversificator.py --experiment exp2 --cadence 900 --max_time 7200