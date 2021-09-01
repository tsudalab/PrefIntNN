import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torch.utils.data as Data

from tqdm import tqdm

import numpy as np

from MLint.utilities.evaluate import ndcg
from MLint.utilities.pairgenerator import pair_generate
from MLint.utilities.plot import plot_acc,plot_error,plot_loss

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

        self.batchnorm = nn.BatchNorm1d(1)

    def forward(self,x1,x2=None):

        h1 = F.relu(self.l1(x1))
        h2 = F.relu(self.l2(h1))
        h = self.l3(h2)
        #h = self.batchnorm(h)
        if x2 is None:
            return h
        else:
            h1_ = F.relu(self.l1(x2))
            h2_ = F.relu(self.l2(h1_))
            h_ = self.l3(h2_)
            #h_ = self.batchnorm(h_)

            p1_sum = torch.exp(torch.sum(h)/len(x1))
            p2_sum = torch.exp(torch.sum(h_)/len(x2))
            p1 = p1_sum/torch.add(p1_sum, p2_sum)
            p2 = p2_sum / torch.add(p1_sum, p2_sum)
            return torch.stack([p1, p2])
        
class PairwiseLoss(object):
    def __init__(self,n_input,n_nodes1,n_nodes2):
        self.n_input = n_input
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2
        self.batch_size = 1
        self.n_epoch = 200
        
        self._create_model()
        
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        
    def loadtraindata(self,xi,xj,yi,yj):
        xi = torch.from_numpy(xi).float()
        xj = torch.from_numpy(xj).float()
        xi, xj = Variable(xi), Variable(xj)
        yi = torch.from_numpy(yi).float()
        yj = torch.from_numpy(yj).float()
        yi, yj = Variable(yi), Variable(yj)
        
        self.x_train = xi
        self.y_train = yi
        
        self.dataset = Data.TensorDataset(xi,xj,yi,yj)
        self.loader = Data.DataLoader(
                            dataset = self.dataset,
                            batch_size = self.batch_size,
                            shuffle = True, num_workers=2,)
    def loadtestdata(self,xi,xj,yi,yj):
        xi_test = torch.from_numpy(xi).float()
        xj_test = torch.from_numpy(xj).float()
        self.xi_test, self.xj_test = Variable(xi_test), Variable(xj_test)
        yi_test = torch.from_numpy(yi).float()
        yj_test = torch.from_numpy(yj).float()
        self.yi_test, self.yj_test = Variable(yi_test), Variable(yj_test)
        
        self.x_test = xi_test
        self.y_test = yi_test     
    def _create_model(self):
        self.model = Model(self.n_input,self.n_nodes1,self.n_nodes2)
        self.criterion = nn.functional.binary_cross_entropy
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.05)
        
    def predict(self,obs,obs2=None):
        self.model.eval()
        with torch.no_grad():
            if obs2 is not None:
                pred = self.model(obs,obs2)
            else:
                pred = self.model(obs)
        return pred

    def load_train_prefs(self,prefs):
        self.preferences = prefs

    def train(self):
        self.model.train(True)
        pref_dist = np.zeros([2], dtype=np.float32)
        for epoch in tqdm(range(self.n_epoch)):
            running_loss = 0

            for step, (batch_xi, batch_xj, batch_yi, batch_yj) in enumerate(self.loader):
                
                b_xi = Variable(batch_xi)
                b_xj = Variable(batch_xj)
                
                if batch_yi < batch_yj:
                    pref_dist[1] = 1
                elif batch_yi > batch_yj:
                    pref_dist[0] = 1
                else:
                    pref_dist[:] = 0.5

                y = torch.from_numpy(pref_dist)

                y_hat = self.model(b_xi,b_xj)

                #print (y_hat)
                
                loss =self.criterion(y_hat, y)
                #print (loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            #print (running_loss/batchsize)
            self.train_loss.append(running_loss/len(self.loader))
            print (self.train_loss[-1])
            pred_y_train = self.predict(self.x_train)
            self.train_acc.append(ndcg(np.c_[self.x_train.numpy(),self.y_train.numpy()], pred_y_train.numpy()))
            
            pred_dist = self.predict(self.xi_test,self.xj_test)
           
            #self.test_loss.append(self.criterion(pred_yi_test,pred_yj_test))
            #self.test_acc.append(ndcg(np.c_[self.x_test.numpy(),self.y_test.numpy()], pred_yi_test.numpy()))
        #plot_loss(self.train_loss,self.test_loss)
        #plt.show()
        #plot_acc(self.train_acc,self.test_acc)
        #plt.show()
            
