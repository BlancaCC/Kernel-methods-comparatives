all: 
	experiments
install:
	bash ./installation.sh
create_dir: 
	mkdir ./results ./results/verboses/ ./results/accuracy_time_stats ./results/joblib
	
experiments: create_dir  
	bash run_experiments.sh

load_to_queu: create_dir  
	sbatch -A gaa_serv -p gaa -c 5 run_experiments.sh
