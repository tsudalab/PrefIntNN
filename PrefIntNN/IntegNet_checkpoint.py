import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torch.utils.data as Data

import numpy as np

from tqdm import tqdm

from MLint.utilities.evaluate import mae,mae,r2,ndcg
from MLint.utilities.pairgenerator import pair_generate
from MLint.utilities.plot import plot_acc,plot_error,plot_loss,plot_R2

import matplotlib.pyplot as plt

import time

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

        h1 = F.dropout(F.relu(self.l1(x)),p=0.15)
        h2 = F.dropout(F.relu(self.l2(h1)),p=0.15)
        h = self.l3(h2)

        return h

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

class IntegNet(object):
    def __init__(self,n_input,n_nodes1,n_nodes2,optim='Adam',batch_size=100, n_epoch=200,learning_rate=0.005,cuda = True):
        self.n_input = n_input
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2

        self.optim = optim
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.cuda = cuda

        if cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        self._create_model()
        
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        self.train_error = []
        self.test_error = []
        self.train_ndcg = []
        self.test_ndcg = []

        self.train_p_loss = []
        self.test_p_loss = []

    def _create_model(self):
        self.model = Model(self.n_input,self.n_nodes1,self.n_nodes2)
        self.model.to(self.device)

        if self.optim == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
        elif self.optim == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),lr=self.learning_rate)
        else:
            raise TypeError('Only Adam and SGD optimizer are supported now!')

    def predict(self,obs):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(obs)
        return pred

    def load_train_normal(self,x,y,batch_size = None):        #Load the dataset with both x and y values which you want to use for fine-tuning
        if batch_size:
            self.batch_size = batch_size

        #self.n_input = x.shape[1]
        self.x_train_n, self.y_train_n = Variable(torch.from_numpy(x).float()), Variable(torch.from_numpy(y).float())
        
        #x, y = Variable(torch.from_numpy(x).float()), Variable(torch.from_numpy(y).float())
        self.dataset = Data.TensorDataset(self.x_train_n,self.y_train_n)
        self.normal_loader = Data.DataLoader(
                            dataset = self.dataset,
                            batch_size = self.batch_size,
                            shuffle = True, num_workers=2,)
    def load_test_normal(self,x_test,y_test):     #Load test dataset with x and y values for validation or test
        self.x_test_n = Variable(torch.from_numpy(x_test).float()).to(self.device)
        self.y_test_n = Variable(torch.from_numpy(y_test).float()).to(self.device)

    def load_train_pair(self,prefs,batch_size = None):
        xi,xj,yi,yj = prefs[0],prefs[1],prefs[2],prefs[3]

        if batch_size:
            self.batch_size = batch_size

        xi,xj = Variable(torch.from_numpy(xi).float()), Variable(torch.from_numpy(xj).float())
        yi,yj = Variable(torch.from_numpy(yi).float()), Variable(torch.from_numpy(yj).float())
        
        self.x_train_p = xi.to(self.device)
        self.y_train_p = yi.to(self.device)
        
        self.dataset = Data.TensorDataset(xi,xj,yi,yj)
        self.pair_loader = Data.DataLoader(
                            dataset = self.dataset,
                            batch_size = self.batch_size,
                            shuffle = True, num_workers=2,)

    def load_test_pair(self,prefs):

        xi,xj,yi,yj = prefs[0],prefs[1],prefs[2],prefs[3]

        self.xi_test,self.xj_test = Variable(torch.from_numpy(xi).float()).to(self.device), Variable(torch.from_numpy(xj).float()).to(self.device)
        self.yi_test,self.yi_test = Variable(torch.from_numpy(yi).float()).to(self.device), Variable(torch.from_numpy(yj).float()).to(self.device)
        
        self.x_test_p = self.xi_test.to(self.device)
        self.y_test_p = self.yi_test.to(self.device)
      
    def train_normal(self,ep = None, lr = None, printres = True, plotres=True):

        self.criterion = nn.MSELoss()
        self.model.train(True)

        if ep:
            self.n_epoch = ep

        if lr:
            for p in self.optimizer.param_groups:
                p['lr'] = lr

        for epoch in tqdm(range(self.n_epoch)):

            running_loss = 0
            running_error = 0
            running_acc = 0


            for step, (batch_x, batch_y) in enumerate(self.normal_loader):

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                
                pred_y = self.model(b_x)
                #print (pred_y)
                
                loss = self.criterion(pred_y, b_y)
                #print (pred_y,b_y)
                error = mae(pred_y.detach().cpu().numpy(),b_y.detach().cpu().numpy())
                acc = r2(b_y.detach().cpu().numpy(),pred_y.detach().cpu().numpy())

                #print (loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_acc += acc
                running_loss += loss.item()
                running_error += error
    

            self.train_loss.append(running_loss / len(self.normal_loader))
            self.train_error.append(running_error/len(self.normal_loader))
            self.train_acc.append(running_acc/len(self.normal_loader))
            #print(f"Training loss: {running_loss/len(self.loader)}")

            pred_y_test = self.predict(self.x_test_n)
            self.test_loss.append(self.criterion(pred_y_test, self.y_test_n))
            self.test_error.append(mae(self.y_test_n.cpu(),pred_y_test.cpu()))
            self.test_acc.append(r2(self.y_test_n.cpu(),pred_y_test.cpu()))      

        if self.cuda:
            torch.cuda.synchronize()                                    #time_ed
        eped = time.time()

        if printres:
            print ('Final Train loss is:', self.train_loss[-1],'Test loss is:', self.test_loss[-1])
            print ('Final Train MAE is:', self.train_error[-1],'Final Test MAE is:', self.test_error[-1])
            print ('Final Train R^2 Score is', self.train_acc[-1],'Final Test R^2 Score is', self.test_acc[-1])
        if plotres:
            plot_loss(self.train_loss,self.test_loss)
            plt.show()
            plot_error(self.train_error,self.test_error)
            plt.show()
            plot_R2(self.train_acc,self.test_acc)
            plt.show()


    def train_pair(self,ep = None, lr = None,printres = True, plotres=True):
        self.model.train(True)
        self.criterion = Pair_loss()

        if ep:
            self.n_epoch = ep

        if lr:
            for p in self.optimizer.param_groups:
                p['lr'] = lr

        for epoch in tqdm(range(self.n_epoch)):
            running_loss = 0

            for step, (batch_xi, batch_xj, batch_yi, batch_yj) in enumerate(self.pair_loader):

                batch_xi, batch_xj, batch_yi, batch_yj = batch_xi.to(self.device), batch_xj.to(self.device), batch_yi.to(self.device), batch_yj.to(self.device)
                
                b_xi = Variable(batch_xi)
                b_xj = Variable(batch_xj)
                
                pred_yi = self.model(b_xi)
                pred_yj = self.model(b_xj)
                
                loss =self.criterion(pred_yi, pred_yj)
        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            #print (running_loss/batchsize)
            self.train_p_loss.append(running_loss/len(self.pair_loader))
            
            pred_y_train = self.predict(self.x_train_p)
            self.train_ndcg.append(ndcg(np.c_[self.x_train_p.cpu().numpy(),self.y_train_p.cpu().numpy()], pred_y_train.cpu().numpy()))
            
            pred_yi_test = self.predict(self.xi_test)
            pred_yj_test = self.predict(self.xj_test)
            
            self.test_p_loss.append(self.criterion(pred_yi_test,pred_yj_test))
            self.test_ndcg.append(ndcg(np.c_[self.x_test_p.cpu().numpy(),self.y_test_p.cpu().numpy()], pred_yi_test.cpu().numpy()))
        if printres:
            print ('Final Train loss is:', self.train_p_loss[-1],'Test loss is:', self.test_p_loss[-1])
            print ('Final Train NDCG is:', self.train_ndcg[-1],'Final Test NDCG is:', self.test_ndcg[-1])
        if plotres:
            plot_loss(self.train_p_loss,self.test_p_loss)
            plt.show()
            plot_acc(self.train_ndcg,self.test_ndcg)
            plt.show()

    def fix_layers(self,n_layers):

        if n_layers == 1:
            for param in self.model.l1.parameters():
                param.requires_grad = False
        if n_layers == 2:
            for param in self.model.l1.parameters():
                param.requires_grad = False
            for param in self.model.l2.parameters():
                param.requires_grad = False














#class Model(nn.Module):
#    def __init__(self,n_layers,n_input,n_nodes,drop_rate=None):
#        super(Model,self).__init__()
#
#        self.n_layers = n_layers    #int
#        self.n_input = n_input      #int, the number of data features/descriptors
#        self.n_nodes = n_nodes      #list-type, length equals n_layers
#
#        if drop_rate:               #list-type, length equals n_layers
#            self.drop_rate = drop_rate
#            if n_layers != len(drop_rate):
#                raise ValueError('Numbers of layers and dropout rate does not match!')
#
#        if n_layers != len(n_nodes):
#            raise ValueError('Numbers of layers and length of n_nodes does not match!')
#
#        layers = []
#        in_nodes = self.n_input
#
#        for i in range(n_layers):
#            layers.append(nn.Linear(in_nodes,n_nodes[i]))
#            layers.append(nn.ReLU())
#            if drop_rate:
#                layers.append(nn.Dropout(drop_rate[i]))
#
#            in_nodes = n_nodes[i]
#
#        layers.append(nn.Linear(in_nodes,1))        #regression, single output node with no activation function
#
#        self.layers = nn.Sequential(*layers)
#
#    def forward(self,x):  
#        return self.layers(x)


