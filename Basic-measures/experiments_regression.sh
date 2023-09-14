#!/bin/sh
dataset=abalone
#dataset=cpusmall
cv=5
njobs=8

echo " run regression models" "$(date)"
python3  run_regression_models.py "$dataset" --cv "$cv" --n_jobs "$njobs"
echo "end training" "$(date)"

