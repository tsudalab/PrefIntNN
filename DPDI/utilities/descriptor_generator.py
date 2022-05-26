import numpy as np
import pandas as pd


from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors, Descriptors

class DescriptorGenerator():

    def __init__(self,df):
        self.df = df
        self.featurelist = []

    def _feature_name(self,df):
        featurelist = list(df.columns)
        return featurelist[3:]   
         
    def rdkit_1d(self, returnfeatures=True):
        col_s,col_p,col_ID = self.df.columns

        Dm = Descriptors.DescriptorCalculator()
        Descriptor_list = []
        for D in Chem.Descriptors._descList:
            Descriptor_list.append(D[0])
        #print (len(Descriptor_list),Descriptor_list)
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(Descriptor_list) 

        smiles = []
        for s in list(self.df[col_s]):
            smiles.append(Chem.MolFromSmiles(s))

        des_values = []
        for mol in smiles:
            pattern = calculator.CalcDescriptors(mol)
            des_values.append(pattern)

        df_descrpitors = pd.DataFrame(des_values,columns=Descriptor_list)
        self.df = pd.concat([self.df,df_descrpitors],axis=1,sort=False)
        self.df.rename(columns={col_s:'smiles',col_p:'value',col_ID:'assay'},inplace=True)
        
        if returnfeatures:
            self.featurelist = self._feature_name(self.df)
        return self.df

    # def _feature_name(self):
    #     featurelist = list(df.columns)
    #     return self.featurelist[3:]