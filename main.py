import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize, rosen, rosen_der,curve_fit, minimize
from MLTools import *       



data_sheet = pd.read_csv("iris.csv")
classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"] #Species

def classify_value(value: str, mark: str) -> int:
    if(value == mark):
        return 1
    else:
        return -1
    
def dataset_classificator(s):
    if(s == "Iris-setosa"):
        return 1
    else:
        return -1
    

data = data_sheet[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]]

data_setosa                = data.copy()
data_setosa["Species"]     = data_setosa["Species"].map(lambda x: classify_value(x,"Iris-setosa"))
data_versicolor            = data.copy()
data_versicolor["Species"] = data_versicolor["Species"].map(lambda x: classify_value(x,"Iris-versicolor"))
data_virginica             = data.copy()
data_virginica["Species"]  = data_virginica["Species"].map(lambda x: classify_value(x,"Iris-virginica"))


def data_sampler(n,k,data):
    ss = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
    s_x = ss[n]
    s_y = ss[k]
    s_c = "Species"
    x = list(data[s_x])
    y = list(data[s_y])
    c = list(data[s_c])
    return (x,y,c)

x_setosa    , y_setosa    , c_setosa     = data_sampler(0,1,data_setosa)
x_versicolor, y_versicolor, c_versicolor = data_sampler(0,1,data_versicolor)
x_virginica , y_virginica , c_virginica  = data_sampler(0,1,data_virginica)

fig, ax = plt.subplots()
minimization_function0 = lambda func, args: minimize(func, x0 = (0,0,0),args=args, method='trust-constr', tol=1e-10).x

def find_classification_line(ax,X,Y,C,color,minimization_function):
    nx,ny,b = minimization_function(min_function2D,(X,Y,C))
    plot_classes2D(ax,X,Y,C,nx,ny,b,color)
    
find_classification_line(ax,x_setosa,y_setosa,c_setosa,"red",minimization_function0)
find_classification_line(ax,x_versicolor,y_versicolor,c_versicolor,"green",minimization_function0)
find_classification_line(ax,x_virginica,y_virginica,c_virginica,"blue",minimization_function0)

ax.grid()
plt.show()