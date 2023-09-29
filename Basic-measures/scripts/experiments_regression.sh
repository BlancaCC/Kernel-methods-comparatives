#!/bin/bash

source ./bcano_python_ml/bin/activate
cv=5
njobs=3
#njobs=8
# Lista de nombres de conjuntos de datos

datasets=( "abalone" "CPU_SMALL" )


# Itera sobre la lista de nombres de conjuntos de datos
for dataset in "${datasets[@]}"; do
    echo " run regression models" "$(date)"
    python3  run_regression_models.py "$dataset" --cv "$cv" --n_jobs "$njobs"
    echo "end training" "$(date)"
done




