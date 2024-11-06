import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import scipy.stats

def label_encode(df: pd.DataFrame, column_name: str, regex: str) -> pd.DataFrame:
    unique_classes = df[column_name].unique()
    for clss in unique_classes:
        name = re.search(regex, clss).group()
        clss_name = name.title()
        df[clss_name] = df[column_name].map(lambda x: 1 if x == clss else -1)    
    return df.drop(column_name,axis=1)

def columns_to_onep_vector(df: pd.DataFrame, column_names: list,column_name: str)-> pd.DataFrame:
    df["One"] = np.ones(df.shape[0])
    column_names.append("One")
    df[column_name] = df[column_names].values.tolist()
    for name in column_names:
        df = df.drop(name,axis=1)
    return df


df = pd.read_csv("iris.csv")
df = df.drop("Id", axis=1)
df = label_encode(df,"Species","[a-zA-Z]*$")
df = columns_to_onep_vector(df,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"],"Features")
df = df.sample(frac=1)

train_df = df.iloc[:120,:]
test_df = df.iloc[120:150,:]
df = df.reset_index()

def mean_error(features: float, target: float, normal: list)-> float:
    return (1-target*np.dot(features, normal))
        
def mse_loss(normal: list, objs: list, class_name: str) -> float:
    summ = 0
    for index, obj in objs.iterrows():
        target = obj[class_name]
        features = obj[-1]
        summ += mean_error(features, target, normal)**2
    return summ

    
     
setosa_normal     = scipy.optimize.minimize(mse_loss, x0 = np.ones(5), args=(train_df,"Setosa"), method='trust-constr', tol=1e-10).x
versicolor_normal = scipy.optimize.minimize(mse_loss, x0 = np.ones(5), args=(train_df,"Versicolor"), method='trust-constr', tol=1e-10).x
virginica_normal  = scipy.optimize.minimize(mse_loss, x0 = np.ones(5), args=(train_df,"Virginica"), method='trust-constr', tol=1e-10).x

setosa_classify     = lambda features: float(np.dot(features, setosa_normal))
versicolor_classify = lambda features: float(np.dot(features, versicolor_normal))
virginica_classify  = lambda features: float(np.dot(features, virginica_normal))


def classify(features: list) -> int:
    predictions = []
    predictions.append(setosa_classify(features))
    predictions.append(versicolor_classify(features))
    predictions.append(virginica_classify(features))
    print(predictions)
    return np.argmax(predictions)

  

train_df["Prediction"] = train_df["Features"].map(classify)
test_df["Prediction"] = test_df["Features"].map(classify)
train_positive = 0
test_positive = 0

for index, obj in train_df.iterrows():
    prediction_index = obj["Prediction"]
    if(obj[prediction_index] == 1):
        train_positive+=1
        
for index, obj in test_df.iterrows():
    prediction_index = obj["Prediction"]
    if(obj[prediction_index] == 1):
        test_positive+=1



print(train_positive/120)   
print(test_positive/30)    
#Accuracy > 0.86
#Precision > 0.87
#Recall > 0.86
