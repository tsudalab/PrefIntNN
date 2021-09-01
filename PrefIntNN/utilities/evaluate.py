import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def mse(y, y_pred):
    return mean_squared_error(y, y_pred)
def mae(y, y_pred):
    return mean_absolute_error(y, y_pred)
def r2(y,y_pred):
    return r2_score(y, y_pred)

def my_old_ndcg(test, mu, printall=0):
    """
    This is to evaluate the ranking accuracy of the test data using 
    Normalized Discounted Cumulative Gain.
    """
    newtest = np.c_[test[:,:-1],mu]
    n = len(test)

    datasort = np.c_[newtest[:,-1],test]
    datasort = datasort[np.lexsort(-datasort.T)]
    ra = np.arange(1,n+1)
    datasort = np.c_[datasort,ra]
    datasort = np.c_[datasort,datasort[:,0]]
    datasort = datasort[np.lexsort(-datasort.T)]
    predrank = datasort[:,-2]
    c = len(predrank)
    idcg = 0
    dcg = 0

    for i in range (1,c+1):
        idcg += (c - i)/(np.log(i+1))
        dcg += (c - predrank[i-1])/(np.log(i+1))
    error = dcg/idcg



    if printall==1:
        print ('predicted ranking is', predrank)

        print ('Normalized Discounted Cumulative Gain is', error)

    return error
def my_new_ndcg(y_true,y_pred):

    def dcg(y_true,y_score):
        n = len(y_score)
        true_rank = np.argsort(y_true)[::-1]
        pred_rank = np.argsort(y_score)[::-1]
        dcg = 0
        for i in range(len(true_rank)):
            nu = n - true_rank[i] - 1
            de = np.log(pred_rank[i] + 2)
            dcg += nu/de
        return dcg

    return dcg(y_true,y_pred)/dcg(y_true,y_true)

def ndcg(y_true, y_score, k=20): #default k = 20
    y_true = y_true.ravel()
    #print (y_true)
    y_score = y_score.ravel()
    #print (y_score)
    y_true_sorted = sorted(y_true, reverse=True)
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
    dcg = 0
    argsort_indices = np.argsort(y_score)[::-1]
    for i in range(k):
        dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
    ndcg = dcg / ideal_dcg
    return ndcg

# def get_dcg(y_pred, y_true, k=20):
#     #注意y_pred与y_true必须是一一对应的，并且y_pred越大越接近label=1(用相关性的说法就是，与label=1越相关)
#     df = pd.DataFrame({"y_pred":y_pred, "y_true":y_true})
#     df = df.sort_values(by="y_pred", ascending=False)  # 对y_pred进行降序排列，越排在前面的，越接近label=1
#     df = df.iloc[0:k, :]  # 取前K个
#     dcg = (2 ** df["y_true"] - 1) / np.log2(np.arange(1, df["y_true"].count()+1) + 1) # 位置从1开始计数
#     dcg = np.sum(dcg)
#     return dcg
# def get_ndcg(y_pred,y_ture, k=20):
#     df = pd.DataFrame({"y_pred":y_pred, "y_true":y_true})
#     df = df.sort_values(by="y_pred", ascending=False)  # 对y_pred进行降序排列，越排在前面的，越接近label=1
#     df = df.iloc[0:k, :]  # 取前K个
#     # df包含y_pred和y_true
#     dcg = get_dcg(df["y_pred"], df["y_true"], k)
#     print(dcg)
#     idcg = get_dcg(df["y_true"], df["y_true"], k)
#     print(idcg)
#     ndcg = dcg / idcg
#     return ndcg

#from sklearn.metrics import ndcg_score
#
#def ndcg(y_true,y_pred):
#    y_true = y_true.reshape(1,-1)
#    y_pred = y_pred.reshape(1,-1)
#    true_rank = np.arange(1,len(y_true))
#    y_pred = np.
#    return ndcg_score(y_true,y_pred)
#
