# Decomposition measurement    

We aim to measure the time used for: 

1. Generate random components (theoretical complexity): for Random Fourier Features and Nyström features
2. Training time for SVM 

1. Use linear SVM versus SVC with linear kernel. 


Requirements

Arguments: 
- Dataset:  X and Y.
- Time to repeat measurement. 
- Transformer of the data (gamma and C are known from the kernel method are in a json file). 
- Learning methods. 
- List of n-component to measure. 

Structure: 
- Json with hyperparamenters. 



