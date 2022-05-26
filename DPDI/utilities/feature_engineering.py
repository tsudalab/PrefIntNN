import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class FeatureEngineer():

    def __init__(self,df,dropcol=['smiles','value','assay']):
        self.df = df 
        self.featurelist = []
        self.dropcol = dropcol

    def _low_std(self,des_df,thd):
  
        des_df = des_df.loc[:,des_df.std()>thd]

        return des_df

    def _high_correlation(self,des_df,thd):

        corr_matrix = des_df.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than 0.9
        to_drop = [column for column in upper.columns if any(upper[column] > thd)]

        # Drop features 
        des_df.drop(to_drop, axis=1, inplace=True)

        return des_df

    def filter(self,std_thd=0.1,corr_thd=0.9,returnfeatures=True):

        #Normalize features and pvalue with minmax scaler
        self.scaler = MinMaxScaler()

        des_df = self.df.drop(self.dropcol,axis=1)
        des_df[:] = self.scaler.fit_transform(des_df[:])
        
        des_df = self._low_std(des_df,std_thd)
        des_df = self._high_correlation(des_df,corr_thd)
        self.df = pd.concat([self.df[self.dropcol],des_df],axis=1,sort=False)

        if returnfeatures:
            self.featurelist = self._feature_name(self.df)
        return self.df

    def _feature_name(self,df):
        featurelist = list(df.columns)
        return featurelist[len(self.dropcol):] 

    def normalize(self,path=None,scaler='MinMax',dropcol = ['smiles','assay']):
        cols_to_norm = self.df.drop(dropcol,axis=1).columns
        df_minmax = pd.DataFrame({'min':self.df[cols_to_norm].min(),'max':self.df[cols_to_norm].max()})
        self.minmax = df_minmax.T
        if path:
            self.minmax.to_csv(path,index=False)  

        self.df[cols_to_norm] =  self.df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return self.df

def reverse(test_set,scaler='MinMax'):

    minmax = pd.read_csv('Data/minmax.csv')
    cols_to_norm = test_set.drop(['smiles','assay'],axis=1).columns
    for col in cols_to_norm:
        test_set[col] = test_set[col].apply(lambda x: (x - minmax[col][0]) / (minmax[col][1] - minmax[col][0]))
    return test_set