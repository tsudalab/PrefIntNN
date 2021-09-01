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

    def forward(self,x):

        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = self.l3(h2)

        return h
    
class Pair_loss(nn.Module):
    def __init__(self):
        super(Pair_loss,self).__init__()
    def forward(self,pred_yi, pred_yj):

        s_ij = 0 #因为我的load的数据集把小的数字放在前面，大的放在后面，所以loss这里只考虑了xi < xj。。。
        if len(pred_yi) > 1: #一个batch中每组数据的loss相加取平均作为整个个batch的loss
            self.loss = 0
            for k in range(len(pred_yi)):
                diff = pred_yi[k] - pred_yj[k]
                self.loss += -s_ij * diff / 2. + torch.log(1 + torch.exp(diff))
            return self.loss/len(pred_yi)
        else:
            diff = pred_yi - pred_yj
            self.loss = -s_ij * diff / 2. + torch.log(1 + torch.exp(diff))
            return self.loss   
            
        
class PairwiseLoss(object):
    def __init__(self,n_input,n_nodes1,n_nodes2):
        self.n_input = n_input
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2
        self.batch_size = 128 #瞎设的
        self.n_epoch = 200
        
        self._create_model()
        
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        
    def loadtraindata(self,xi,xj,yi,yj): #这里xi，xj，yi，yj都是通过pair_generate过来的，具体写在了pairgenerator里面
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
        self.criterion = Pair_loss()
        self.optimizer = optim.SGD(self.model.parameters(),lr=0.005) #Adam我也试了 都没好使，lr也改过，weightdecay也加过 没好使
        
    def predict(self,obs):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(obs)
        return pred

    def train(self):
        self.model.train(True)

        for epoch in tqdm(range(self.n_epoch)): 
            running_loss = 0

            for step, (batch_xi, batch_xj, batch_yi, batch_yj) in enumerate(self.loader):
                
                b_xi = Variable(batch_xi)
                b_xj = Variable(batch_xj)
                
                pred_yi = self.model(b_xi)
                pred_yj = self.model(b_xj)
                
                loss =self.criterion(pred_yi, pred_yj) 
                #这个prefs的模型，每次要训练俩个数据才能得到loss
                #根据paper的公式（3），这两个数据的差值oij会影响loss
                #我发现优化的过程中，nn会把这几个预测值变得越来越小越来越小，于是越来越接近-log2，就卡在0.69
                #但是我加了batchnorm在最后的输出层也没好使
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            #print (running_loss/batchsize)
            self.train_loss.append(running_loss/len(self.loader))
            print ('loss:',self.train_loss[-1])
            pred_y_train = self.predict(self.x_train)
            self.train_acc.append(ndcg(np.c_[self.x_train.numpy(),self.y_train.numpy()], pred_y_train.numpy()))
            
            pred_yi_test = self.predict(self.xi_test)
            pred_yj_test = self.predict(self.xj_test)
            
            self.test_loss.append(self.criterion(pred_yi_test,pred_yj_test))
            self.test_acc.append(ndcg(np.c_[self.x_test.numpy(),self.y_test.numpy()], pred_yi_test.numpy()))
        plot_loss(self.train_loss,self.test_loss)
        plt.show()
        plot_acc(self.train_acc,self.test_acc)
        plt.show()
            
