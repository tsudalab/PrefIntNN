import optuna

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torch.utils.data as Data

from tqdm import tqdm

from DPDI.utilities.evaluate import mse,mae,ndcg
from DPDI.utilities.pairs_generator import PairsGenerator

def choose_data(df):
	#Choose your main group for tuning. Otherwise choose the assay group with largest molecule set
	assay_n = 0
	for assay in df['assay'].unique():
	    if len(df[df['assay']==assay]) > assay_n:
	        assay_n = len(df[df['assay']==assay])
	        assay_group = assay
	    
	df = df[df['assay']==assay_group]
	return df

def loadtraindata(batch_size):
    xi_train = torch.from_numpy(xi).float()
    xj_train = torch.from_numpy(xj).float()
    xi_train, xj_train = Variable(xi_train), Variable(xj_train)
    yi_train = torch.from_numpy(yi).float()
    yj_train = torch.from_numpy(yj).float()
    yi_train, yj_train = Variable(yi_train), Variable(yj_train)


    dataset = Data.TensorDataset(xi_train,xj_train,yi_train,yj_train)
    loader = Data.DataLoader(
                        dataset = dataset,
                        batch_size = batch_size,
                        shuffle = True, num_workers=2,)
    return loader

def define_model(trial,n_input):
    #optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int('n_layers', 1,3)# the number of layers from 1 to 3
    layers =[]
    
    in_features = n_input
    for i in range(n_layers):
        
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 1024)
        layers.append(nn.Linear(in_features,out_features))
        layers.append(nn.ReLU())
        
        p = trial.suggest_float("dropout_l{}".format(i), 0, 0.4)
        layers.append(nn.Dropout(p))
        
        in_features = out_features
    layers.append(nn.Linear(in_features,1))
    
    return nn.Sequential(*layers)

class Pair_loss(nn.Module):
    def __init__(self):
        super(Pair_loss,self).__init__()
    def forward(self,pred_yi, pred_yj):
        s_ij = -1 #xi < xj
        if len(pred_yi) > 1:
            self.loss = 0
            for k in range(len(pred_yi)):
                diff = pred_yi[k] - pred_yj[k]
                self.loss += (1 - s_ij) * diff / 2. + torch.log(1 + torch.exp(-diff))
            return self.loss/len(pred_yi)
        else:
            diff = pred_yi - pred_yj
            self.loss = (1 - s_ij) * diff / 2. + torch.log(1 + torch.exp(-diff))
            return self.loss 


def objective(trial):
    model = define_model(trial, n_input)
    
    #generate optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam","SGD"])
    lr = trial.suggest_float('lr',1e-5,1e-1,log = True)
    batch_size = trial.suggest_categorical('batchsize',[16,32,64,128,256,512])
    optimizer = getattr(optim, optimizer_name)(model.parameters(),lr=lr)
    
    train_loader = loadtraindata(batch_size)
    criterion = Pair_loss()
    
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for step, (batch_xi, batch_xj, batch_yi, batch_yj) in enumerate(train_loader):
                
                b_xi = Variable(batch_xi)
                b_xj = Variable(batch_xj)
                
                pred_yi = model(b_xi)
                pred_yj = model(b_xj)
                
                loss = criterion(pred_yi, pred_yj)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        model.eval()
        #x_test, y_test = loadtestdata()
        with torch.no_grad():
            pred_yi_test = model(xi_test)
            pred_yj_test = model(xj_test)
            
            #print (y_test,pred_y_test)
        acc = ndcg(y_test.numpy(), pred_yi_test.numpy().flatten(),k=50)
        trial.report(acc,epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return acc

if __name__ == '__main__':

	df = pd.read_csv('Data/chembl_fax_rdkit1d_normalized.csv')
	df.head()

	df = choose_data(df)

	train_set, test_set = train_test_split(df,test_size = 0.2,shuffle= True, random_state=1)
	PG = PairsGenerator(train_set)
	train_prefs = PG.generate()
	PG = PairsGenerator(test_set)
	test_prefs = PG.generate()

	xi,xj,yi,yj = train_prefs
	xi_test,xj_test,yi_test,yj_test = test_prefs

	xi_test = torch.from_numpy(xi_test).float()
	xj_test = torch.from_numpy(xj_test).float()
	xi_test, xj_test = Variable(xi_test), Variable(xj_test)
	yi_test = torch.from_numpy(yi_test).float()
	yj_test = torch.from_numpy(yj_test).float()
	yi_test, yj_test = Variable(yi_test), Variable(yj_test)

	x_test = xi_test
	y_test = yi_test  

	n_input = xi.shape[1]

	n_epochs = 50

	study = optuna.create_study(direction='maximize')
	study.optimize(objective, n_trials =30, timeout=None)

	pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
	complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))
        
	with open('model/hyperparams.txt', 'w') as f:
		print("Study statistics: ",file=f)
		print("  Number of finished trials: ", len(study.trials),file=f)
		print("  Number of pruned trials: ", len(pruned_trials),file=f)
		print("  Number of complete trials: ", len(complete_trials),file=f)

		print("Best trial:",file=f)
		trial = study.best_trial

		print("  Value: ", trial.value,file=f)

		print("  Params: ",file=f)
		for key, value in trial.params.items():
			print("    {}: {}".format(key, value),file=f)
		f.close()