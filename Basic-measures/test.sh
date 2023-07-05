dataset=a9a
cv=2
njobs=10

echo "Basic-measures/model_Nystrom_svm.py classification" "$(date -utc)"
python3 model_rbf_svm_classification.py "$dataset" --cv "$cv" --n_jobs "$njobs" 
echo "end training" "$(date -utc)"

