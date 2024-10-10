
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


#plane: (n,x) = 0
def plane_projection2D(x,y,nx,ny):
    return np.dot((nx,ny),(x,y))

def classifier2D(x,y,nx,ny):
    return np.sign(plane_projection2D(x,y,nx,ny))

def min_function2D(g,X,Y,C):
    def l_MSE(nx,ny,b, x,y,c):
        return (1 - c*plane_projection2D(x, y-b, nx, ny))**2
    
    nx,ny,b = g
    summ = 0
    for i in range(len(X)):
        summ += l_MSE(nx,ny,b, X[i],Y[i], C[i] )
    return summ

def plot_classes2D(X,Y,C,nx,ny,b):
    X_1 = []
    Y_1 = []
    X1 = []
    Y1 = []
    for i in range(len(X)):
        if(C[i] == 1):
            X1.append(X[i])
            Y1.append(Y[i])
        if(C[i] == -1):
            X_1.append(X[i])
            Y_1.append(Y[i])


    fig, ax = plt.subplots()
    ax.plot(X1,  Y1-b,  linestyle = "none", color = "red",  marker="s")
    ax.plot(X_1, Y_1-b, linestyle = "none", color = "blue", marker="s")
    
    
    x_space = np.linspace(min(X), max(X), 100)
    #ax.set_xlim(min(X)*(1-0.1),max(X)*(1+0.1))
    #ax.set_ylim(min(Y-b)*(1-0.1),max(Y-b)*(1+0.1))
    
    ax.plot(x_space, -nx*x_space/ny, linestyle = "--", color = "black", marker="")

    ax.grid()