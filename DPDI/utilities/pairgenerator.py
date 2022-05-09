import numpy as np
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

def data_split(X,y,val_size = 0.2,test_size = None,random_state = False,normalize = True):
    n = random_state
    if not test_size:
        val = val_size   
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val,random_state=n,shuffle = True)

        if normalize:
            #normalizer = preprocessing.StandardScaler()
            normalizer = preprocessing.MinMaxScaler()
            X_train = normalizer.fit_transform(X_train)
            X_val = normalizer.fit_transform(X_val)
            y_train = normalizer.fit_transform(y_train)
            y_val = normalizer.fit_transform(y_val)
            return X_train, X_val, y_train, y_val
        return X_train, X_val, y_train, y_val

    else:
        size1 = val_size + test_size
        size2 = test_size/size1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size1,shuffle = True)
        #print (X_train,type(X_train))
        X_val, X_test, y_val, y_test = train_test_split(X_test,y_test,test_size = size2, shuffle = True)

        if normalize:
            #normalizer = preprocessing.StandardScaler()
            normalizer = preprocessing.MinMaxScaler()
            X_train = normalizer.fit_transform(X_train)
            X_val = normalizer.fit_transform(X_val) 
            y_train = normalizer.fit_transform(y_train)
            y_val = normalizer.fit_transform(y_val)
            X_test = normalizer.fit_transform(X_test)
            y_test = normalizer.fit_transform(y_test)
            return X_train, X_val, X_test,y_train, y_val,y_test
        return X_train, X_val, X_test,y_train, y_val,y_test





def pair_generate(xydata,reverse=False,normalize=False): 

    if len(xydata) <= 1:
        print ('INSUFFICIENT DATA')
        return -1
    else:
        normalizer = preprocessing.StandardScaler()
        preferences = []
        if reverse:
            data = xydata[np.argsort(xydata[:,-1])][::-1] 
        else:
            data = xydata[np.argsort(xydata[:,-1])] #order ascending
         #keep only features, no target values
        if normalize:
            data = normalizer.fit_transform(data)

        for i in range(len(data)-1):
            if i == 0:
                xi = data[i,:-1]
                xj = data[i+1,:-1]
                yi = data[i,-1]
                yj = data[i+1,-1]
            else:
                preferences.append([data[i],data[i+1]])
                xi = np.vstack((xi,data[i,:-1]))
                xj = np.vstack((xj,data[i+1,:-1]))
                yi = np.vstack((yi,data[i,-1]))
                yj = np.vstack((yj,data[i+1,-1]))
        return [xi,xj,yi,yj]# a < b 

def random_pair_generate(xydata,reverse=False,n_pairs = 2000): #这个是我怕全是小于关系不能训练，于是大于小于随机了一下，但是是后来改的，输出形式不一样，暂时不能用
    preferences = []
    n = len(xydata)

    x1_ind = np.random.randint(0,n,n_pairs,dtype='int')
    x2_ind = np.random.randint(0,n,n_pairs,dtype='int')

    for i in range(n_parirs):
        x1 = xydata[x1_ind,:-1]
        x2 = xydata[x2_ind,:-1]
        y1 = xydata[x1_ind,-1]
        y2 = xydata[x2_ind,-1]

        if y1 > y2:
            preferences.append([x1,x2,0])
        elif y1 < y2:
            preferences.append([x1,x2,1])
        else:
            preferences.append([x1,x2,0.5])

def full_generator(dataset,reverse = False,normalize=False): #按小于排列
    if normalize:
        normalizer = preprocessing.StandardScaler()
        dataset = normalizer.fit_transform(dataset)
    x1 = []
    x2 = []
    y1= []
    y2 = []
    n = len(dataset)
    x = dataset[:,:-1]
    y = dataset[:,-1]
    for i in range(n):
        for j in range(i,n):
            if y[i] < y[j]:
                x1.append(x[i])
                x2.append(x[j])
                y1.append(y[i])
                y2.append(y[j])
            elif y[i] > y[j]:
                x1.append(x[j])
                x2.append(x[i])
                y1.append(y[j])
                y2.append(y[i])
    if not reverse:
        return [np.array(x1),np.array(x2),np.array(y1),np.array(y2)]
    else:
        return [np.array(x2),np.array(x1),np.array(y2),np.array(y1)]

def add_pairs(prefs,newprefs):
    prefs[0].append(newprefs[0]) #x1
    prefs[1].append(newprefs[1]) #x2
    prefs[2].append(newprefs[2]) #x3
    prefs[3].append(newprefs[3]) #x4
    return prefs 



