#!/bin/sh
dataset=Diabetes
#dataset=a9a
cv=5
njobs=8

echo " run classificaiton models" "$(date)"
python3  run_classification_models.py "$dataset" --cv "$cv" --n_jobs "$njobs" 
echo "end training" "$(date)"

