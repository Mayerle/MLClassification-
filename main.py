import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import scipy
import scipy.optimize
from instruments.stats import *
from instruments.dftools import *
from instruments.functions import *
from instruments.view import *

class LCModel:
    
    def __init__(self, train: ClassificationDF, test: ClassificationDF):
        self.train = train
        self.test = test
        self.__normals = {}
        
    def fit(self, class_index: int, loss_function) -> list:
        classes = self.train.get_classes(class_index)
        features = self.train.get_features()
        features_length = self.train.get_features_length()
        
        normal = self.__minimize(loss_function,features_length,features,classes)
        self.__normals[class_index] = normal
        return normal
    
    def __minimize(self,loss_function, dimention, features, classes) -> list:
        x0 = np.zeros(dimention)
        args = (features,classes)
        method="trust-constr"
        tol=10**(-10)
        return scipy.optimize.minimize(loss_function, x0 = x0, args = args,method=method, tol=tol).x
    
    #def predict(self):
        
        
        

df = pd.read_csv("datasets/iris.csv")
train_df, _, test_df = sample_df(df)
class_names = ["","Setosa","Versicolor","Virginica"]

#Train dataframe
editor_train = FeatureEditor(train_df.iloc[:,1:-1])
editor_train.standardization_normalize()
editor_train.convert_to_onevectors()

encoder_train = Encoder(train_df.iloc[:,-1])
encoder_train.label_encode()


#Test dataframe
editor_test = FeatureEditor(test_df.iloc[:,1:-1])
editor_test.standardization_normalize()
editor_test.convert_to_onevectors()

encoder_test = Encoder(test_df.iloc[:,-1])
encoder_test.label_encode()


#Classification dataframes
cdf_train = ClassificationDF(editor_train, encoder_train)
cdf_test = ClassificationDF(editor_test, encoder_test)

cdf_train.encoder.label_encode()
c = LCModel(cdf_train,cdf_test)

result = c.fit(0,logistic_loss)

print(result)
#print(cdf_train.get_features())





