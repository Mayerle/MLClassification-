import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize, rosen, rosen_der,curve_fit, minimize
from MLTools import *       

def get_zero_array(n: int) -> list:
    return [0 for x in range(n)]

def classify_value(value, mark) -> int:
    if(value == mark):
        return 1
    else:
        return -1

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
    return (presicion,recall)


# 2D

def plane_projection2D(x,y,nx,ny):
    return np.dot((nx,ny),(x,y))

def classifier2D(x,y,nx,ny,b):
    return np.sign(plane_projection2D(x,y-b,nx,ny))

def min_function2D_MSE(params,X,Y,C):
    
    def l_MSE(x,y,c, nx,ny,b):
        return (1 - c*plane_projection2D(x, y-b, nx, ny))**2
    #def l_MSE(nx,ny,b, x,y,c):
    #    return (1 - c*plane_projection2D(x, y-b, nx, ny))**2
    
    nx,ny,b = params
    summ = 0
    for i in range(len(X)):
        summ += l_MSE(X[i],Y[i], C[i], nx, ny, b)
    return summ

def find_classification_line(X,Y,C):
    nx,ny,b =  minimize(min_function2D_MSE, x0 = (0,0,0),args=(X,Y,C), method='trust-constr', tol=1e-10).x
    return (nx,ny,b)

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

# 3D

def plane_projection3D(x,y,z,nx,ny,nz):
    return np.dot((nx,ny,nz),(x,y,z))

def classifier3D(x,y,z,nx,ny,nz,b):
    return np.sign(plane_projection3D(x,y,z-b,nx,ny,nz))

def min_function3D_MSE(params,X,Y,Z,C):
    
    def l_MSE(x,y,z,c, nx,ny,nz,b):
        return (1 - c*plane_projection3D(x, y,z-b, nx, ny,nz))**2
    #def l_MSE(nx,ny,b, x,y,c):
    #    return (1 - c*plane_projection2D(x, y-b, nx, ny))**2
    
    nx,ny,nz,b = params
    summ = 0
    for i in range(len(X)):
        summ += l_MSE(X[i],Y[i],Z[i], C[i], nx, ny,nz, b)
    return summ

def find_classification_plane(X,Y,Z,C):
    nx,ny,nz,b =  minimize(min_function3D_MSE, x0 = (0,0,0,0),args=(X,Y,Z,C), method='trust-constr', tol=1e-10).x
    return (nx,ny,nz,b)

def plot_classes3D(ax,nx,ny,nz,b,color,X,Y,Z,C):
    x = []
    y = []
    z = []
    for i in range(len(C)):
        if(C[i] == 1):
            x.append(X[i])
            y.append(Y[i])
            z.append(Z[i])
            
    ax.scatter(x,y,z,color = color,marker="s")
    number_points = 5
    x_space = np.linspace(min(X), max(X), number_points)
    y_space = np.linspace(min(Y), max(Y), number_points)
    x_space, y_space = np.meshgrid(x_space, y_space)
    z_space = -nx*x_space/nz-ny*y_space/nz+b
    ax.plot_surface(x_space, y_space, z_space ,color = color, alpha=0.2)
    
    ax.set_zlim(min(Z), max(Z))

# nD

def plane_projectionND(x,n):
    return np.dot(x,n)

def classifierND(x,n,b):
    x[len(x)-1]-=b
    return np.sign(plane_projectionND(x,n))

def min_functionND_MSE(params,x,c):
    
    def l_MSE(vector,clss, n,b):
        print(vector)
        vector[len(vector)-1]=vector[len(vector)-1] - b
        return (1 - clss*plane_projectionND(vector, n))**2
    
    n = params[:-1]
    b = params[-1]
    
    summ = 0
    for i in range(len(x)):
        summ += l_MSE(x[i], c[i], n, b)
    print(summ)
    return summ

def find_classification_hyperplane(x,c):
    dimension = len(x)
    x0 = get_zero_array(dimension+1)
    opts =  minimize(min_functionND_MSE, x0 = x0, args=(x,c), method='trust-constr', tol=1e-10).x
    n = opts[:-1]
    b = opts[-1]
    return [n,b]

