import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize, rosen, rosen_der,curve_fit, minimize
from MLTools import *  
import statistics as st

def dataframe_to_list_of_lists(dataframe):
    rows = []
    count = dataframe.shape[0]
    for i in range(count):
        row = list(dataframe.iloc[i,:])
        rows.append(row)
    return rows

def data_to_vectors_and_classes(data):
    vectors = []
    classes = []
    rows = len(data)
    
    for row in range(rows):
        vc = data[row]
        vector = vc[:-1]
        clss = vc[-1]
        vectors.append(vector)
        classes.append(clss)
            
    return [vectors,classes]

def stat_data(objects,targets,normal,intersection):
    classificator = lambda x: classifierND(x,normal,intersection)
    accuracy = calculate_accuracy(objects,targets,classificator)
    precision, recall = calculate_precision_recall(objects,targets,classificator)
    accuracies.append(accuracy)
    precisions.append (precision)
    recalls.append (recall)

data_sheet = pd.read_csv("iris.csv")
data_frame = data_sheet.rename({"Species": "Class"}, axis='columns')
data_frame = data_frame[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Class"]]


df_setosa              = data_frame.copy()
df_versicolor          = data_frame.copy()
df_virginica           = data_frame.copy()

df_setosa["Class"]     = df_setosa["Class"].map(lambda x: classify_value(x,"Iris-setosa"))
df_versicolor["Class"] = df_versicolor["Class"].map(lambda x: classify_value(x,"Iris-versicolor"))
df_virginica["Class"]  = df_virginica["Class"].map(lambda x: classify_value(x,"Iris-virginica"))


data_setosa     = dataframe_to_list_of_lists(df_setosa)
data_versicolor = dataframe_to_list_of_lists(df_versicolor)
data_virginica  = dataframe_to_list_of_lists(df_virginica)

setosa = data_to_vectors_and_classes(data_setosa)
versicolor = data_to_vectors_and_classes(data_versicolor)
virginica  = data_to_vectors_and_classes(data_virginica)


accuracies = []
precisions = []
recalls = []

normal, intersection = find_classification_hyperplane(min_functionND_MSE,*setosa)
stat_data(*setosa,normal,intersection)

normal, intersection = find_classification_hyperplane(min_functionND_MSE,*versicolor)
stat_data(*versicolor,normal,intersection)

normal, intersection = find_classification_hyperplane(min_functionND_MSE,*virginica)
stat_data(*virginica,normal,intersection)

#Accuracy > 0.86
#Precision > 0.87
#Recall > 0.86
print(f"[MSE]\n{st.mean(accuracies):.2f} | {st.mean(precisions):.2f} | {st.mean(recalls):.2f}")


accuracies = []
precisions = []
recalls = []

normal, intersection = find_classification_hyperplane(min_functionND_logistic,*setosa)
stat_data(*setosa,normal,intersection)

normal, intersection = find_classification_hyperplane(min_functionND_logistic,*versicolor)
stat_data(*versicolor,normal,intersection)

normal, intersection = find_classification_hyperplane(min_functionND_logistic,*virginica)
stat_data(*virginica,normal,intersection)

#Accuracy > 0.86
#Precision > 0.87
#Recall > 0.86
print(f"[Logistic]\n{st.mean(accuracies):.2f} | {st.mean(precisions):.2f} | {st.mean(recalls):.2f}")