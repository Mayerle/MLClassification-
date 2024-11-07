import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import scipy.stats
from npfunctions import *
RANDOM_STATE = 3

df = pd.read_csv("iris.csv")
df = df.drop("Id", axis=1)
df = label_encode(df,"Species","[a-zA-Z]*$")
df = columns_to_onep_vector(df,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"],"Features")
df = df.sample(frac=1,random_state=RANDOM_STATE)

train_df = df.iloc[:120,:]
test_df = df.iloc[120:150,:]
df = df.reset_index()

setosa_normal     = scipy.optimize.minimize(mse_loss, x0 = np.ones(5), args=(train_df,"Setosa"), method='trust-constr', tol=1e-10).x
versicolor_normal = scipy.optimize.minimize(mse_loss, x0 = np.ones(5), args=(train_df,"Versicolor"), method='trust-constr', tol=1e-10).x
virginica_normal  = scipy.optimize.minimize(mse_loss, x0 = np.ones(5), args=(train_df,"Virginica"), method='trust-constr', tol=1e-10).x
#setosa_normal     = [ 0.03694015,  0.5451582,  -0.39111431, -0.13786857, -0.58188164]
#versicolor_normal = [ 0.14571849, -1.05602803,  0.28970243, -0.89657659,  2.01034151]
#virginica_normal  = [-0.18265992,  0.51086951,  0.10141165,  1.03444675, -2.42845239]
normals = [setosa_normal, versicolor_normal, virginica_normal]



classificator = get_classificator(normals)

train_df["Prediction"] = train_df["Features"].map(classificator)
test_df["Prediction"] = test_df["Features"].map(classificator)

            
print("[MSE-OneVsRest Classification]\n")     
print("[Train dataset metrics]\n")
print_stats(train_df)
print("\n\n[Test dataset metrics]\n")
print_stats(test_df)











df = pd.read_csv("iris.csv")
df = df.drop("Id", axis=1)
df = label_encode(df,"Species","[a-zA-Z]*$")
df = columns_to_onep_vector(df,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"],"Features")
df = df.sample(frac=1,random_state=RANDOM_STATE)

train_df = df.iloc[:120,:]
test_df = df.iloc[120:150,:]
df = df.reset_index()

setosa_normal     = scipy.optimize.minimize(logistic_loss, x0 = np.ones(5), args=(train_df,"Setosa"), method='trust-constr', tol=1e-10).x
versicolor_normal = scipy.optimize.minimize(logistic_loss, x0 = np.ones(5), args=(train_df,"Versicolor"), method='trust-constr', tol=1e-10).x
virginica_normal  = scipy.optimize.minimize(logistic_loss, x0 = np.ones(5), args=(train_df,"Virginica"), method='trust-constr', tol=1e-10).x
#setosa_normal     = [ 0.03694015,  0.5451582,  -0.39111431, -0.13786857, -0.58188164]
#versicolor_normal = [ 0.14571849, -1.05602803,  0.28970243, -0.89657659,  2.01034151]
#virginica_normal  = [-0.18265992,  0.51086951,  0.10141165,  1.03444675, -2.42845239]
normals = [setosa_normal, versicolor_normal, virginica_normal]



classificator = get_classificator(normals)

train_df["Prediction"] = train_df["Features"].map(classificator)
test_df["Prediction"] = test_df["Features"].map(classificator)

            
print("\n\n[Logistic-OneVsRest Classification]\n")      
print("[Train dataset metrics]\n")
print_stats(train_df)
print("\n\n[Test dataset metrics]\n")
print_stats(test_df)

mse_conf_matrix = calculate_confusion_matrix(train_df)
show_confusion_matrix(mse_conf_matrix)
logistic_conf_matrix = calculate_confusion_matrix(train_df)
show_confusion_matrix(logistic_conf_matrix)
plt.show()








"""
df = pd.read_csv("iris.csv")
df = df.drop("Id", axis=1)
df = label_encode(df,"Species","[a-zA-Z]*$")
df = columns_to_onep_vector(df,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"],"Features")
df = df.sample(frac=1,random_state=RANDOM_STATE)

train_df = df.iloc[:120,:]
test_df = df.iloc[120:150,:]
df = df.reset_index()

setosa_normal     = scipy.optimize.minimize(mse_loss, x0 = np.ones(5), args=(train_df,"Setosa"), method='trust-constr', tol=1e-10).x


versicolor_normal = scipy.optimize.minimize(mse_loss, x0 = np.ones(5), args=(train_df[train_df["Setosa"] == -1],"Versicolor"), method='trust-constr', tol=1e-10).x
virginica_normal  = scipy.optimize.minimize(mse_loss, x0 = np.ones(5), args=(train_df[train_df["Setosa"] == -1],"Virginica"), method='trust-constr', tol=1e-10).x
#setosa_normal     = [ 0.03694015,  0.5451582,  -0.39111431, -0.13786857, -0.58188164]
#versicolor_normal = [ 0.14571849, -1.05602803,  0.28970243, -0.89657659,  2.01034151]
#virginica_normal  = [-0.18265992,  0.51086951,  0.10141165,  1.03444675, -2.42845239]
normals = [setosa_normal, versicolor_normal, virginica_normal]



classificator = get_classificator(normals)

train_df["Prediction"] = train_df["Features"].map(classificator)
test_df["Prediction"] = test_df["Features"].map(classificator)

            
print("[MSE-OneVsRest-OneVsOne Classification]\n")     
print("[Train dataset metrics]\n")
print_stats(train_df)
print("\n\n[Test dataset metrics]\n")
print_stats(test_df)

plt.show()
#Accuracy > 0.86
#Precision > 0.87
#Recall > 0.86
"""