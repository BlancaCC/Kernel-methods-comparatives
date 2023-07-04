#!/bin/sh

# Check if the number of njobs argument is provided
if [ $# -ge 1 ]; then
  njobs=$1
else
  njobs=5
fi

# Check if the dataset name argument is provided
if [ $# -ge 2 ]; then
  dataset=$2
else
    echo "Arguments must be: run_experiment njobs datasets"
    echo "For example: ./run_experiments 10 a9a"
    exit 1
fi

if [ $# -ge 3 ]; then
  cv=$3
else
   cv=5
fi

# List of valid datasets
valid_datasets=("a9a")  # Add more valid datasets if needed

# Check if the dataset is valid
if ! [[ " ${valid_datasets[*]} " == *" ${dataset} "* ]]; then
  echo "The provided dataset is not valid. Valid datasets are:"
  printf '%s\n' "${valid_datasets[@]}"
  exit 1
fi
TZ='Europe/Madrid'

source ./bcano_python_ml/bin/activate
echo "Modelo KVSM" "$(date -u)"
python3 model_ksvm.py "$dataset" --cv "$cv" --n_jobs "$njobs" > ./results/verboses/ksvm_"$dataset"_cv_5.txt
echo ""
echo "Modelo Nystrom "$(date -u)"
python3 model_Nystrom_ridge_classification.py "$dataset" --cv "$cv" --n_jobs "$njobs" > ./results/verboses/Nystrom_ridge_classification_"$dataset"_cv_5.txt
echo ""
echo "Modelo rbf  "$(date -u)"
python3 model_rbf_ridge_classification.py "$dataset" --cv "$cv" --n_jobs "$njobs" > ./results/verboses/rbf_ridge_classification_a9a_cv_5.txt