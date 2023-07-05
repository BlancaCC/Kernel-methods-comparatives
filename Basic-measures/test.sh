dataset=a9a
cv=2
njobs=10
TZ='Europe/Madrid'

echo "Modelo mlp classification" "$(date -u)"
python3 model_mlp_classification.py "$dataset" --cv "$cv" --n_jobs "$njobs" 
echo "Modelo mlp classification" "$(date -u)"