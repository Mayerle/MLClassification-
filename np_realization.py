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
#setosa_normal     = [ 0.03694015,  0.5451582,  -0.39111431, -0.13786857, -0.58188164]
#versicolor_normal = [ 0.14571849, -1.05602803,  0.28970243, -0.89657659,  2.01034151]
#virginica_normal  = [-0.18265992,  0.51086951,  0.10141165,  1.03444675, -2.42845239]

setosa_predict    = lambda features: float(np.dot(features, setosa_normal))
versicolor_predict = lambda features: float(np.dot(features, versicolor_normal))
virginica_predict  = lambda features: float(np.dot(features, virginica_normal))


def predict(features: list) -> int:
    predictions = []
    predictions.append(setosa_predict(features))
    predictions.append(versicolor_predict(features))
    predictions.append(virginica_predict(features))
    return np.argmax(predictions)

  

train_df["Prediction"] = train_df["Features"].map(predict)
test_df["Prediction"] = test_df["Features"].map(predict)
train_positive = 0
test_positive = 0


def calculate_accuracy(df: pd.DataFrame):
    positive = 0
    for _, obj in df.iterrows():
        prediction_index = obj["Prediction"]
        if(obj[prediction_index] == 1):
            positive+=1
    return positive/df.shape[0] 

def calculate_stats(df: pd.DataFrame,class_index: int) -> list:
    df_observation = df.iloc[:,class_index]
    df_prediction = df.iloc[:,-1].map(lambda x: 1 if x == class_index else -1)
    op_df = pd.DataFrame(data={"Observation":df_observation,"Prediction":df_prediction})
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for _, obj in op_df.iterrows():
        prediction  = obj["Prediction"]
        observation = obj["Observation"]
        if(prediction == observation):
            if(prediction == 1):
                tp += 1
            else:
                tn += 1
        else:
            if(prediction == 1):
                fp += 1
            else:
                fn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return [precision,recall]

    
def print_stats(df: pd.DataFrame):
    accuracy = calculate_accuracy(df)
    setosa_stats = calculate_stats(df,0)
    versicolor_stats = calculate_stats(df,1)
    virginica_stats  = calculate_stats(df,2)
    avg_precision = np.mean([virginica_stats[0],setosa_stats[0],versicolor_stats[0]])
    avg_recall = np.mean([virginica_stats[1],setosa_stats[1],versicolor_stats[1]])
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"[Setosa]\nPrecision: {setosa_stats[0]:.2f}\nRecall: {setosa_stats[1]:.2f}\n")
    print(f"[Versicolor]\nPrecision: {versicolor_stats[0]:.2f}\nRecall: {versicolor_stats[1]:.2f}\n")
    print(f"[Virginica]\nPrecision: {virginica_stats[0]:.2f}\nRecall: {virginica_stats[1]:.2f}\n")
    print(f"[Average metrics]\nPrecision: {avg_precision:.2f}\nRecall: {avg_recall:.2f}")
print("\n\n")     
print("[Train dataset metrics]\n")
print_stats(train_df)
print("\n\n[Test dataset metrics]\n")
print_stats(test_df)
#setosa_df
#print(train_df.columns)
#Accuracy > 0.86
#Precision > 0.87
#Recall > 0.86
