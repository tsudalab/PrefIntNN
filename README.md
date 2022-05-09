# PrefIntNN
A neural network for integrating data via pairwise loss function. The PrefIntNN package provides a neural network-based way of integrating data via learning pairwise relations. Please check our paper at: https://pubs.acs.org/doi/10.1021/acsmedchemlett.1c00439.    
This is an extension of our previous Gaussian process-based package [PrefInt](https://github.com/tsudalab/PrefInt).

## Requirements
Python = 3.7
numpy = 1.21.5
pandas = 1.3.4
rdkit = 2020.09.1.0
pytorch = 1.11.0
scikit-learn = 1.0.2

## Installation
To deploy the package, please create or update the environment first with anaconda or miniconda:
```sh
conda env create -f dpdi.yaml

conda env update -n your-env --file dpdi.yml
```
Clone the repository by:
```sh
git clone https://github.com/tsudalab/PrefIntNN.git

cd PrefIntNN
```
Then you are ready to check and run the tutorials notebooks.

## Run DPDI Tutorial
The model requires specific form of data input. For data preparation, please check: [data_preparation_tutorial](https://github.com/tsudalab/PrefIntNN/blob/master/Data_Preparation_Tutorial.ipynb).    

The model is implemented via Pytorch. For training model, please refer to: [Model_Training_Prediction Tutorial](https://github.com/tsudalab/PrefIntNN/blob/master/Model_Training_and_Prediction.ipynb).

## License
The PrefIntNN package is licensed under the MIT "Expat" License.
