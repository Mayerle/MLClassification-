import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import scipy.stats


def one_hot_encode(df: pd.DataFrame, column_name: str, new_column_name: str):
    unique_classes = df[column_name].unique()
    encode =  df[column_name].map(lambda x: np.where(unique_classes == x)[0][0])
    df.insert(5,"Classis",encode)


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

def sigmoid(x):
    return (1+np.exp(-x))**-1
def logistic_loss(normal: list, objs: list, class_name: str) -> float:
    summ = 0
    for index, obj in objs.iterrows():
        target = obj[class_name]
        features = obj[-1]
        margin = target*np.dot(features, normal)
        summ -= np.log(sigmoid(margin))
    return summ

def mean_error(features: list, target: float, normal: list)-> float:
    return (1-target*np.dot(features, normal))

     
def mse_loss(normal: list, objs: list, class_name: str) -> float:
    summ = 0
    for index, obj in objs.iterrows():
        target = obj[class_name]
        features = obj[-1]
        margin = target*np.dot(features, normal)
        summ += mean_error(features, target, normal)**2
    return summ


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
    
    
    
def show_confusion_matrix(matrix: np.ndarray) -> None:
    fig, axe = plt.subplots()
    graph = axe.matshow(matrix,cmap = "Blues")
    axe.set_xlabel("Predicted class")
    axe.set_ylabel("Observed class")
    axe.set_xticklabels(["","Setosa","Versicolor","Virginica"])
    axe.set_yticklabels(["","Setosa","Versicolor","Virginica"])
    fig.colorbar(graph, ax = axe) 
    
    

def calculate_confusion_matrix(df: pd.DataFrame) -> np.ndarray:
    confusion_matrix = np.zeros((3,3))
    for _, obj in df.iterrows():
        prediction = obj["Prediction"]
        observation = np.argmax(obj.iloc[0:3].tolist())
        confusion_matrix[observation,prediction] += 1
    return confusion_matrix

#def predict(features: list) -> int:
    #predictions = []
    #predictions.append(setosa_predict(features))
    #predictions.append(versicolor_predict(features))
    #predictions.append(virginica_predict(features))
    #return np.argmax(predictions)
def get_classificator(normals: list) -> int:
    setosa_predict    = lambda features: float(np.dot(features, normals[0]))
    versicolor_predict = lambda features: float(np.dot(features, normals[1]))
    virginica_predict  = lambda features: float(np.dot(features, normals[2]))
    def classify(features: list) -> int:
        predictions = []
        predictions.append(setosa_predict(features))
        predictions.append(versicolor_predict(features))
        predictions.append(virginica_predict(features))
        return np.argmax(predictions)
    return classify

#setosa_predict    = lambda features: float(np.dot(features, setosa_normal))
#versicolor_predict = lambda features: float(np.dot(features, versicolor_normal))
#virginica_predict  = lambda features: float(np.dot(features, virginica_normal))
