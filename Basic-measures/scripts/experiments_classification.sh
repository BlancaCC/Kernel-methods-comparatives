#!/bin/sh

cv=5
njobs=8

#!/bin/bash

# Lista de nombres de conjuntos de datos
# obviamos a9a por ser muy grande
#datasets=( "Diabetes" "a9a" "covtype.binary")
datasets=("Diabetes")
# Itera sobre la lista de nombres de conjuntos de datos
for dataset in "${datasets[@]}"; do
    echo " run classification models" "$(date)" for dataset "$dataset"
    python3  run_classification_models.py "$dataset" --cv "$cv" --n_jobs "$njobs" 
    echo "end training" "$(date)"
done


