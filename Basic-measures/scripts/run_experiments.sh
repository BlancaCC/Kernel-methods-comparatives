#!/bin/sh

# run: ./run_experiments.sh njobs dataset cv 
# Check if the number of njobs argument is provided
if [ $# -ge 1 ]; then
  njobs=$1
else
  #njobs=5
  njobs=10
fi

# Check if the dataset name argument is provided
if [ $# -ge 2 ]; then
  dataset=$2
else
    #dataset=a9a
    #dataset=covtype.binary
    dataset=Diabetes
fi

if [ $# -ge 3 ]; then
  cv=$3
else
   cv=5
fi

# List of valid datasets
valid_datasets=("a9a" "CPU_SMALL" "Diabetes" "covtype.binary")  # Add more valid datasets if needed

# Check if the dataset is valid
if ! [[ " ${valid_datasets[*]} " == *" ${dataset} "* ]]; then
  echo "The provided dataset is not valid. Valid datasets are:"
  printf '%s\n' "${valid_datasets[@]}"
  exit 1
fi

source ./bcano_python_ml/bin/activate
echo "Modelo KVSM" "$(date -u)"
python3 model_ksvm.py "$dataset" --cv "$cv" --n_jobs "$njobs" > ./results/verboses/ksvm_"$dataset"_cv_"$cv".txt
echo ""
echo "Modelo Nystrom ridge classification." "$(date -u)"
python3 model_Nystrom_ridge_classification.py "$dataset" --cv "$cv" --n_jobs "$njobs" > ./results/verboses/Nystrom_ridge_classification_"$dataset"_cv_"$cv".txt
echo ""
echo "Modelo rbf ridge classification"  "$(date -u)"
python3 model_rbf_ridge_classification.py "$dataset" --cv "$cv" --n_jobs "$njobs" > ./results/verboses/rbf_ridge_classification_"$dataset"_cv_"$cv".txt
echo ""
echo "Modelo mlp classification" "$(date -u)"
python3 model_mlp_classification.py "$dataset" --cv "$cv" --n_jobs "$njobs" > ./results/verboses/mlp_classification_"$dataset"_cv_"$cv".txt
echo ""
echo "Modelo rbf + svm classification" "$(date -u)"
python3 model_rbf_svm_classification.py "$dataset" --cv "$cv" --n_jobs "$njobs" > ./results/verboses/rbf_svm_classification_"$dataset"_cv_"$cv".txt
echo ""
echo "Modelo Nystrom svm" "$(date -u)"
python3 model_Nystrom_svm.py "$dataset" --cv "$cv" --n_jobs "$njobs" > ./results/verboses/Nystrom_svm_classification_"$dataset"_cv_"$cv".txt
echo ""
echo " model_kernel_ridge_classification " "$(date -u)"
python3 model_kernel_ridge_classification.py "$dataset" --cv "$cv" --n_jobs "$njobs" > ./results/verboses/kernel_ridge_classification_"$dataset"_cv_"$cv".txt
echo "end experiments :)" "$(date -u)"