import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize, rosen, rosen_der,curve_fit, minimize
from MLTools import *       

def get_zero_array(n: int) -> list:
    return [0 for x in range(n)]

def find_classification_line(X,Y,C):
    nx,ny,b =  minimize(min_function2D_MSE, x0 = (0,0,0),args=(X,Y,C), method='trust-constr', tol=1e-10).x
    return (nx,ny,b)


def classify_value(value, mark) -> int:
    if(value == mark):
        return 1
    else:
        return -1
#plane: (n,x) = 0
def plane_projection2D(x,y,nx,ny):
    return np.dot((nx,ny),(x,y))

def classifier2D(x,y,nx,ny,b):
    return np.sign(plane_projection2D(x,y-b,nx,ny))


def get_zero_array(n: int) -> list:
    return [0 for x in range(n)]


def min_function2D_MSE(params,X,Y,C):
    
    def l_MSE(nx,ny,b, x,y,c):
        return (1 - c*plane_projection2D(x, y-b, nx, ny))**2
    #def l_MSE(nx,ny,b, x,y,c):
    #    return (1 - c*plane_projection2D(x, y-b, nx, ny))**2
    
    nx,ny,b = params
    summ = 0
    for i in range(len(X)):
        summ += l_MSE(nx,ny,b, X[i],Y[i], C[i])
    return summ

def calculate_accuracy(objects,targets,classifier):
    count = len(targets)
    corrects = 0
    i=0
    for vector in objects:
        if(classifier(vector) == targets[i]):
            corrects+=1
        i+=1
    return corrects/count
    
def calculate_precision_recall(objects,targets,classifier):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    i  = 0
    for vector in objects:
        prediction = classifier(vector)
        target = targets[i]
        if(prediction == target):
            if(prediction == 1):
                tp+=1
            else:
                tn+=1
        else:
            if(prediction == 1):
                fp+=1
            else:
                fn+=1
        i+=1
    presicion = tp/(tp+fp)
    recall = tp/(tp+fn)
    #return (tp,fp,tn,fn)
    return (presicion,recall)

def plot_classes2D(ax,nx,ny,b,color,X,Y,C):
    x = []
    y = []
    for i in range(len(C)):
        if(C[i] == 1):
            x.append(X[i])
            y.append(Y[i])
    ax.plot(x,  y,  linestyle = "none", color = color,  marker="s")
    x_space = np.linspace(min(X), max(X), 100)
    ax.plot(x_space, -nx*x_space/ny+b, linestyle = "--", color = color, marker="")

