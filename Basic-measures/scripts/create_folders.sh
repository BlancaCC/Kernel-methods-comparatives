#!/bin/bash

# Define the path to the "data" directory
data_dir="./Data"
mkdir -p "./results"
# Iterate through the folders in the "data" directory
for folder in "$data_dir"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        
        # Create the subfolders for the current folder
        mkdir -p "./results/$folder_name/"
        mkdir -p "./results/$folder_name/verboses/"
        mkdir -p "./results/$folder_name/accuracy_time_stats/"
        mkdir -p "./results/$folder_name/joblib/"
        mkdir -p "./analysis/$folder_name/"
        mkdir -p "./analysis/$folder_name/latex-tables/"
        mkdir -p "./analysis/$folder_name/plot/"
        
        echo "Created result and analys forlder for $folder_name"
    fi
done

echo "Folder creation completed."
