all: 
	experiments
	create_dir

install:
	bash ./installation.sh

create_dir: 
	mkdir ./results/abalone/ ./results/abalone/verboses/ ./results/abalone/accuracy_time_stats ./results/abalone/joblib 
	mkdir ./results/CPU_SMALL/ ./results/CPU_SMALL/verboses/ ./results/CPU_SMALL/accuracy_time_stats ./results/CPU_SMALL/joblib 
	mkdir ./results ./results/verboses/ ./results/accuracy_time_stats ./results/joblib 
	
	
experiments: create_dir  
	bash run_experiments.sh

load_to_queu: create_dir  
	sbatch -A gaa_serv -p gaa -c 15 run_experiments.sh
clear: 
	bash clear.sh

get_results:
	scp -r bcano@casarrubuelos:/home/bcano/Basic-measures/results .  
