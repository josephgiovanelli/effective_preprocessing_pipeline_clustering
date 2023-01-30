#!/bin/bash
docker stop auto_clues
docker rm auto_clues
docker build -t auto_clues .
docker run --name auto_clues --volume $(pwd):/home --detach -t auto_clues
docker exec auto_clues bash ./scripts/wrapper_experiments.sh