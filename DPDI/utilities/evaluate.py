import numpy as np
import pandas as pd 

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def mse(y, y_pred):
    return mean_squared_error(y, y_pred)
def mae(y, y_pred):
    return mean_absolute_error(y, y_pred)
def r2(y,y_pred):
    return r2_score(y, y_pred)

# ndcg
def _dcg_exp(y_true, y_pred, k):
    #注意y_pred与y_true必须是一一对应的，并且y_pred越大越接近label=1(用相关性的说法就是，与label=1越相关)

    df = pd.DataFrame({"y_pred":y_pred, "y_true":y_true})
    df = df.sort_values(by="y_pred", ascending=False)  # 对y_pred进行降序排列，越排在前面的，越接近label=1
    df = df.iloc[0:k, :]  # 取前K个
    
    dcg = (2 ** df["y_true"] - 1) / np.log2(np.arange(1, df["y_true"].count()+1) + 1) # 位置从1开始计数
    dcg = np.sum(dcg)
    return dcg

def _dcg_linear(y_true,y_pred,k):

    df = pd.DataFrame({"y_pred":y_pred, "y_true":y_true})
    df = df.sort_values(by="y_pred", ascending=False)
    c = len(df['y_true'])
    df = df.iloc[0:k, :]
    dcg = (c - df['y_true']) / np.log2(np.arange(1,  df["y_true"].count()+1) +1 )
    dcg = np.sum(dcg)
    return dcg


def _ndcg(df,form,k):
    if not k:
        k = len(y_true)
    # df包含y_pred和y_true
    if form == 'exponential':
        df.sort_values(by='y_true',inplace=True)
        df['y_true'] = np.arange(1,len(df)+1) #只看排序位置，不看y_true的值,值越大权重越大，数字越大
        
        dcg = _dcg_exp(df["y_true"],df["y_pred"],k)
        idcg = _dcg_exp(df["y_true"],df["y_true"],k)
        ndcg = dcg / idcg
        return ndcg

    elif form == 'linear':
        df.sort_values(by='y_true',inplace=True,ascending=False)
        df['y_true'] = np.arange(1,len(df)+1) #只看排序位置，不看y_true的值，值越大排位越高，数字越小
        dcg = _dcg_linear(df["y_true"],df["y_pred"],k)
        df['y_pred'] = np.array(df['y_true'].sort_values(ascending=False))
        idcg = _dcg_linear(df["y_true"],df["y_pred"],k)
        ndcg = dcg / idcg
        return ndcg

def ndcg(y_true,y_pred,form='linear',k=None):

    df = pd.DataFrame({"y_pred":y_pred, "y_true":y_true})

    if not k:
        k = len(y_true)
    else:
        k = min(len(y_true),k)
    
    return _ndcg(df,form,k)