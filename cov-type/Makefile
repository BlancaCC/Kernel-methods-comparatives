all: 
	experiments
	create_dir

install:
	bash ./scripts/installation.sh

create_dir: 
	bash ./scripts/create_folders.sh
	
	
experiments: create_dir  
	bash ./run_regression_models.py

load_to_queu: create_dir  
	sbatch -A gaa_serv -p gaa -c 15 run_experiments.sh
clear: 
	bash clear.sh

get_results:
	scp -r bcano@casarrubuelos:./Basic_copy/results/ ./
