import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torch.utils.data as Data

import numpy as np
import copy

from tqdm import tqdm

from DPDI.utilities.evaluate import mae,mae,r2,ndcg
from DPDI.utilities.plot import plot_acc,plot_error,plot_loss,plot_R2

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

        #self.batchnorm = nn.BatchNorm1d(1)

    def forward(self,x):

        h1 = F.dropout(F.relu(self.l1(x)),p=0.15)
        h2 = F.dropout(F.relu(self.l2(h1)),p=0.15)
        h = self.l3(h2)
        #h = self.batchnorm(h)

        return h

class Pair_loss(nn.Module):
    def __init__(self):
        super(Pair_loss,self).__init__()
    def forward(self,pred_yi, pred_yj):

        self.loss = 0 
        s_ij = 0 #yi<yj
        for k in range(len(pred_yi)):
            diff = (pred_yi[k] - pred_yj[k])
            self.loss +=  - s_ij * diff + torch.log(1 + torch.exp(diff))

        return self.loss/len(pred_yi)
 

class IntegNet(object):
    def __init__(self,n_input,n_nodes1,n_nodes2,optim='Adam',weight_decay = 0,batch_size=128, n_epoch=200,learning_rate=0.001,cuda = 'cuda:0'):
        self.n_input = n_input
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2

        self.optim = optim
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.cuda = cuda
        self.weight_decay = weight_decay

        if cuda:
            self.device = torch.device(cuda)
        else:
            self.device = torch.device('cpu')
        
        self._create_model()
        
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.train_error = []
        self.val_error = []
        self.train_ndcg = []
        self.val_ndcg = []

        self.train_p_loss = []
        self.val_p_loss = []


    def _create_model(self):
        self.model = Model(self.n_input,self.n_nodes1,self.n_nodes2)
        self.model.to(self.device)

        if self.optim == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),weight_decay = self.weight_decay,lr=self.learning_rate)
        elif self.optim == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),weight_decay = self.weight_decay,lr=self.learning_rate)
        else:
            raise TypeError('Add your own optimizer here!')

    def predict(self,obs):
        self.model.eval()

        if type(obs) == np.ndarray:
            obs = Variable(torch.from_numpy(obs).float()).to(self.device)
            with torch.no_grad():
                pred = self.model(obs)
            return np.array(pred.flatten().cpu()) 

        with torch.no_grad():
            pred = self.model(obs)
        return pred.flatten()

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
    def load_val_normal(self,x_val,y_val):     #Load val dataset with x and y values for validation or val
        self.x_val_n = Variable(torch.from_numpy(x_val).float()).to(self.device)
        self.y_val_n = Variable(torch.from_numpy(y_val).float()).to(self.device)

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

    def load_val_pair(self,prefs,val_x,val_y):

        xi,xj,yi,yj = prefs[0],prefs[1],prefs[2],prefs[3]

        self.xi_val,self.xj_val = Variable(torch.from_numpy(xi).float()).to(self.device), Variable(torch.from_numpy(xj).float()).to(self.device)
        self.yi_val,self.yj_val = Variable(torch.from_numpy(yi).float()).to(self.device), Variable(torch.from_numpy(yj).float()).to(self.device)
        

        self.x_val_p = Variable(torch.from_numpy(val_x).float()).to(self.device)
        self.y_val_p = Variable(torch.from_numpy(val_y).float()).to(self.device)
      
    def train_normal(self,ep = None, lr = None, n_epochs_stop = None,printres = True, plotres=True):

        self.criterion = nn.MSELoss()
        self.model.train(True)

        min_val_loss = 1e+6
        epochs_no_imporve = 0

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

            pred_y_val = self.predict(self.x_val_n)
            self.val_loss.append(self.criterion(pred_y_val, self.y_val_n))
            self.val_error.append(mae(self.y_val_n.cpu(),pred_y_val.cpu()))
            self.val_acc.append(r2(self.y_val_n.cpu(),pred_y_val.cpu()))      

            #early stop
            if n_epochs_stop is not None:
                val_loss = self.val_loss[-1]
                #print(val_loss)
                if val_loss < min_val_loss:
                    #Saving the model
                    self.best_model = copy.deepcopy(self.model.state_dict())
                    epochs_no_imporve = 0
                    min_val_loss = val_loss
                else:
                    epochs_no_imporve += 1
                    if epochs_no_imporve == n_epochs_stop:
                        print ('Early Stopping at Epoch:',epoch - n_epochs_stop)
                        self.model.load_state_dict(self.best_model)
                        break


        if printres:
            print ('Final Train loss is:', self.train_loss[-1],'val loss is:', self.val_loss[-1])
            print ('Final Train MAE is:', self.train_error[-1],'Final val MAE is:', self.val_error[-1])
            print ('Final Train R^2 Score is', self.train_acc[-1],'Final val R^2 Score is', self.val_acc[-1])
        if plotres:
            plot_loss(self.train_loss,self.val_loss.cpu())
            plt.show()
            plot_error(self.train_error,self.val_error)
            plt.show()
            plot_R2(self.train_acc,self.val_acc)
            plt.show()


    def train_pair(self,ep = None, lr = None, n_epochs_stop = None,printres = True, plotres=True,printloss = False):
        self.model.train(True)
        self.criterion = Pair_loss()

        min_val_loss = 1e+6
        epochs_no_imporve = 0

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
            if printloss:
                print (self.train_p_loss[-1])
            
            pred_y_train = self.predict(self.x_train_p)
            
            self.train_ndcg.append(ndcg(self.y_train_p.cpu().numpy(), pred_y_train.cpu().numpy()))
            
            pred_yi_val = self.predict(self.xi_val)
            pred_yj_val = self.predict(self.xj_val)
            pred_y_val = self.predict(self.x_val_p)
            
            self.val_p_loss.append(self.criterion(pred_yi_val,pred_yj_val).item())
            if printloss:
                print (self.val_p_loss[-1])
            self.val_ndcg.append(ndcg(self.y_val_p.cpu().numpy(), pred_y_val.cpu().numpy()))

            #early stop
            if n_epochs_stop is not None:

                val_loss = self.val_p_loss[-1]
                #print (val_loss)
                if val_loss < min_val_loss:
                    #Saving the model
                    self.best_model = copy.deepcopy(self.model.state_dict())
                    epochs_no_imporve = 0
                    min_val_loss = val_loss
                else:
                    epochs_no_imporve += 1
                    if epochs_no_imporve == n_epochs_stop:
                        print ('Early Stopping at Epoch:',epoch - n_epochs_stop)
                        self.model.load_state_dict(self.best_model)
                        break

        if printres:
            print ('Final Train loss is:', self.train_p_loss[-1],'val loss is:', self.val_p_loss[-1])
            print ('Final Train NDCG is:', self.train_ndcg[-1],'Final val NDCG is:', self.val_ndcg[-1])
        if plotres:
            plot_loss(self.train_p_loss,self.val_p_loss)
            plt.show()
            plot_acc(self.train_ndcg,self.val_ndcg)
            plt.show()


    def predict_test(self,test_X,test_y):

        test_X = Variable(torch.from_numpy(test_X).float()).to(self.device)
        pred_y = self.predict(test_X)
        self.true_rank_para = test_y
        self.pred_rank_para = pred_y.cpu()
        acc = ndcg(np.c_[test_X.cpu().numpy(),test_y],pred_y.cpu().numpy())
        return acc



    def fix_layers(self,n_layers):

        if n_layers == 1:
            for param in self.model.l1.parameters():
                param.requires_grad = False
        if n_layers == 2:
            for param in self.model.l1.parameters():
                param.requires_grad = False
            for param in self.model.l2.parameters():
                param.requires_grad = False

















