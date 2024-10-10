import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize, rosen, rosen_der,curve_fit, minimize
from MLTools import *       



data_sheet = pd.read_csv("iris.csv")
classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"] #Species

def dataset_classificator(s):
    if(s == "Iris-setosa"):
        return 1
    else:
        return -1
    

x = data_sheet[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]]
x = x[(x["Species"] == "Iris-setosa") | (x["Species"] == "Iris-versicolor")]
x["Species"] = x["Species"].map(dataset_classificator)

def data_sampler(n,k):
    ss = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]
    s_x = ss[n]
    s_y = ss[k]
    X = list(x[s_x])
    Y = list(x[s_y])
    return (X,Y)


def find_classification_line(X,Y):
    C = list(x["Species"])
    nx,ny,b = minimize(min_function2D, x0 = (0,0,0),args=(X,Y,C), method='trust-constr', tol=1e-10).x
    print(nx,ny,b)
    plot_classes2D(X,Y,C,nx,ny,b)

find_classification_line(*data_sampler(0,1))
find_classification_line(*data_sampler(0,2))
find_classification_line(*data_sampler(0,3))
find_classification_line(*data_sampler(1,2))
find_classification_line(*data_sampler(1,3))
find_classification_line(*data_sampler(2,3))

plt.show()