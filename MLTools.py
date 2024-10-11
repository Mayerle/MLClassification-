
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


#plane: (n,x) = 0
def plane_projection2D(x,y,nx,ny):
    return np.dot((nx,ny),(x,y))

def classifier2D(x,y,nx,ny):
    return np.sign(plane_projection2D(x,y,nx,ny))

def min_function2D(params,X,Y,C):
    
    def l_MSE(nx,ny,b, x,y,c):
        return (1 - c*plane_projection2D(x, y-b, nx, ny))**2
    
    nx,ny,b = params
    summ = 0
    for i in range(len(X)):
        summ += l_MSE(nx,ny,b, X[i],Y[i], C[i])
    return summ


def plot_classes2D(ax,X,Y,C,nx,ny,b,color):
    x = []
    y = []
    for i in range(len(C)):
        if(C[i] == 1):
            x.append(X[i])
            y.append(Y[i])
    ax.plot(x,  y,  linestyle = "none", color = color,  marker="s")
    x_space = np.linspace(min(X), max(X), 100)
    ax.plot(x_space, -nx*x_space/ny+b, linestyle = "--", color = color, marker="")

