# Deep Preference Data Integration (DPDI)
DPDI package provides a neural network-based method of integrating data via learning pairwise relations. 
For integrating outcome variables of different assay types, a surrogate variable is introduced, and a neural network is trained such that the total order induced by the surrogate variable is maximally consistent with given data sets. By integrating multiple data sets into surrogate values, the number of samples used in statistical analysis is increased, leading to more conclusive results.

<img src="https://pubs.acs.org/cms/10.1021/acsmedchemlett.1c00439/asset/images/large/ml1c00439_0006.jpeg" width=90% height=90%>

For details, please check our paper at: https://pubs.acs.org/doi/10.1021/acsmedchemlett.1c00439.       

This is an extension of our previous Gaussian process-based package [PrefInt](https://github.com/tsudalab/PrefInt).

## Requirements
Python = 3.7    
numpy = 1.21.5    
pandas = 1.3.4    
rdkit = 2020.09.1.0    
pytorch = 1.11.0    
scikit-learn = 1.0.2    

## Installation

Clone the repository by:
```sh
git clone https://github.com/tsudalab/PrefIntNN.git

cd PrefIntNN
```

To deploy the package, please create or update the environment with anaconda or miniconda:
```sh
conda env create -f dpdi.yaml
```
or
```
conda env update -n your-env --file dpdi.yml
```
Activate the enviroment:
```sh
conda activate dpdi
```

Then you are ready to check and run the tutorials notebooks.

## DPDI running Tutorials
The model requires specific form of data input. For data preparation, please check: [Data_Preparation_tutorial](https://github.com/tsudalab/PrefIntNN/blob/master/Tutorials/Data_Preparation_Tutorial.ipynb).    

The model is implemented via Pytorch. For training model, please refer to: [Model_Training_Prediction Tutorial](https://github.com/tsudalab/PrefIntNN/blob/master/Tutorials/Model_Training_and_Prediction.ipynb).

A hyperparamters tuning script via [Optuna](https://optuna.org/) is also provided.

## License
The PrefIntNN package is licensed under the MIT "Expat" License.

## Contact
Xiaolin SUN (xlsun324@gmail.com)
