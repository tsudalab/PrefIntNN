import numpy as np
import pandas as pd

class PairsGenerator():

    def __init__(self,df):
        self.df = df
        self.prefs = []
        self.x = np.array(df.drop(['smiles','value','assay'],axis=1))
        self.y = np.array(df['value'])
        
    def generate(self,verbost=1,normalize=True):

        for assay in self.df['assay'].unique():
            subset = self.df[self.df['assay']==assay]
            prefs_subset = self.full_generator(subset)
            #print (assay,prefs_subset)
            if prefs_subset: #subset data larger than 2 has vlidate subset pairs
                
                if len(self.prefs) == 0: #if first generation
                    self.prefs = prefs_subset

                else:
                    for i in range(4):
                        #print(self.prefs[i],prefs_subset[i])
                        self.prefs[i] = np.r_[self.prefs[i],prefs_subset[i]]
        if verbost == 1:
            print('The number of identical assay group is:',len(self.df['assay'].unique()))
            print('The length of total prefs is:', len(self.prefs[0]))
        return self.prefs

    def full_generator(self,subset,reverse = False): #按小于排列

        x1 = []
        x2 = []
        y1= []
        y2 = []
        n = len(subset)

        if n <= 1:
            #print('Subset data less than 2')
            return 0

        x = np.array(subset.drop(['smiles','value','assay'],axis=1))
        y = np.array(subset['value'])

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

        if not len(x1): #for the case that all molecule has same affinity
            return 0

        if not reverse:
            prefs_subset = [np.array(x1),np.array(x2),np.array(y1),np.array(y2)]
        else:
            prefs_subset = [np.array(x2),np.array(x1),np.array(y2),np.array(y1)]
        return prefs_subset




