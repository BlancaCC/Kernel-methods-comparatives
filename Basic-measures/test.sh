#!/bin/sh
dataset=Diabetes
#dataset=a9a
cv=5
njobs=5

echo " model_kernel_ridge_classification.py" "$(date)"
python3 model_kernel_ridge_classification.py "$dataset" --cv "$cv" --n_jobs "$njobs" > ./results/verboses/kernel_ridge_classification_"$dataset"_cv_"$cv".txt
echo "end training" "$(date)"

