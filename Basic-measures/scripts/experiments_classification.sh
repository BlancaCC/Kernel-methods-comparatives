#!/bin/bash

source ./bcano_python_ml/bin/activate
cv=5
njobs=5
# si estás ejecutando en tu ordenador recuerda que tiene 8 núcleos de eficienca

# Lista de nombres de conjuntos de datos
# obviamos covtype por ser muy grande
#datasets=( "Diabetes" "a9a" )
datasets=( "Diabetes" )
#datasets=( "w3a" "a7a" )
#datasets=( "Diabetes" "w3a" "a5a")

# Itera sobre la lista de nombres de conjuntos de datos
for dataset in "${datasets[@]}"; do
    echo " run classification models" "$(date)" for dataset "$dataset"
    python3  run_classification_models.py "$dataset" --cv "$cv" --n_jobs "$njobs" 
    echo "end training" "$(date)"
done


