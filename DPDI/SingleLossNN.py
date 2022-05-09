import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torch.utils.data as Data

import numpy as numpy

from tqdm import tqdm

from MLint.utilities.evaluate import mae,mae,r2
from MLint.utilities.plot import plot_acc,plot_error,plot_loss,plot_R2

import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self,n_input,n_nodes1,n_nodes2):
        super(Model, self).__init__()

        self.n_input = n_input
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2

        self.l1 = nn.Linear(self.n_input, self.n_nodes1)
        self.l2 = nn.Linear(self.n_nodes1, self.n_nodes2)
        self.l3 = nn.Linear(self.n_nodes2, 1)

    def forward(self,x):

        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = self.l3(h2)

        return h

class SingleLoss(object):
    def __init__(self,n_input,n_nodes1,n_nodes2):
        self.n_input = n_input
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2
        self.batch_size = 100
        self.n_epoch = 50
        self._create_model()
        self.train_loss = []
        self.test_loss = []
        self.train_error = []
        self.test_error = []
        self.train_acc =  []
        self.test_acc = []
    def loadtraindata(self,x,y):
        self.n_input = x.shape[1]
        
        x, y = Variable(x), Variable(y)       
        #x, y = Variable(torch.from_numpy(x).float()), Variable(torch.from_numpy(y).float())
        self.dataset = Data.TensorDataset(x,y)
        self.loader = Data.DataLoader(
                            dataset = self.dataset,
                            batch_size = self.batch_size,
                            shuffle = True, num_workers=2,)
    def loadtestdata(self,x_test,y_test):
        
        self.x_test, self.y_test = Variable(x_test), Variable(y_test)       
        #x, y = Variable(torch.from_numpy(x).float()), Variable(torch.from_numpy(y).float())
        
        
    def _pairloss(self,x_i,x_j,t_i,t_j):
        return 0
        
    def _create_model(self):
        self.model = Model(self.n_input,self.n_nodes1,self.n_nodes2)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(),lr=0.005)
    
    def predict(self,obs):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(obs)
        return pred
    
    def train(self):
        self.model.train(True)
        
        for epoch in tqdm(range(self.n_epoch)):
            running_loss = 0
            running_error = 0
            running_acc = 0
            for step, (batch_x, batch_y) in enumerate(self.loader):
                
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                
                pred_y = self.model(b_x)
                #print (pred_y)
                
                loss = self.criterion(pred_y, b_y)
                
                error = mae(pred_y.detach().numpy(),b_y.detach().numpy())
                acc = r2(b_y.detach().numpy(),pred_y.detach().numpy())
                
                #print (loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_acc += acc
                running_loss += loss.item()
                running_error += error
                
            self.train_loss.append(running_loss / len(self.loader))
            self.train_error.append(running_error/len(self.loader))
            self.train_acc.append(running_acc/len(self.loader))
            #print(f"Training loss: {running_loss/len(self.loader)}")
            
            pred_y_test = self.predict(self.x_test)
            self.test_loss.append(self.criterion(pred_y_test, self.y_test))
            self.test_error.append(mae(pred_y_test,self.y_test))
            self.test_acc.append(r2(pred_y_test,self.y_test))
        print ('Final Train loss is:', self.train_loss[-1],'Test loss is:', self.test_loss[-1])
        plot_loss(self.train_loss,self.test_loss)
        plt.show()
        print ('Final Train MAE is:', self.train_error[-1],'Final Test MAE is:', self.test_error[-1])
        plot_error(self.train_error,self.test_error)
        plt.show()
        print ('Final Train R^2 Score is', self.train_acc[-1],'Final Test R^2 Score is', self.test_acc[-1])
        plot_R2(self.train_acc,self.test_acc)
        plt.show()
            
    