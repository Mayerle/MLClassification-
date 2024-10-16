import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize, rosen, rosen_der,curve_fit, minimize
from MLTools import *       

def sample_data(data):
    x = list(data.iloc[:,0])
    y = list(data.iloc[:,1])
    c = list(data.iloc[:,2])
    return (x,y,c)
#classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"] #Classes
#["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Class"]


data_sheet = pd.read_csv("iris.csv")

data_frame = data_sheet.rename({"Species": "Class"}, axis='columns')
data_frame = data_frame[["SepalLengthCm","SepalWidthCm","Class"]]


df_setosa              = data_frame.copy()
df_versicolor          = data_frame.copy()
df_virginica           = data_frame.copy()

df_setosa["Class"]     = df_setosa["Class"].map(lambda x: classify_value(x,"Iris-setosa"))
df_versicolor["Class"] = df_versicolor["Class"].map(lambda x: classify_value(x,"Iris-versicolor"))
df_virginica["Class"]  = df_virginica["Class"].map(lambda x: classify_value(x,"Iris-virginica"))




setosa     = sample_data(df_setosa)
versicolor = sample_data(df_versicolor)
virginica  = sample_data(df_virginica)



accuracies = []
precisions = []
recalls = []

fig, ax = plt.subplots()

nx,ny,b = find_classification_line(*setosa)
plot_classes2D(ax,nx,ny,b,"red",*setosa)


classificator = lambda xy: classifier2D(xy[0],xy[1],nx,ny,b)
x,y,targets = setosa
objects = list(zip(x,y))
accuracy = calculate_accuracy(objects,targets,classificator)
precision, recall = calculate_precision_recall(objects,targets,classificator)
accuracies.append(accuracy)
precisions.append (precision)
recalls.append (recall)


nx,ny,b = find_classification_line(*versicolor)
plot_classes2D(ax,nx,ny,b,"blue",*versicolor)

classificator = lambda xy: classifier2D(xy[0],xy[1],nx,ny,b)
x,y,targets = versicolor
objects = list(zip(x,y))
accuracy = calculate_accuracy(objects,targets,classificator)
precision, recall = calculate_precision_recall(objects,targets,classificator)
accuracies.append(accuracy)
precisions.append (precision)
recalls.append (recall)

nx,ny,b = find_classification_line(*virginica)
plot_classes2D(ax,nx,ny,b,"green",*virginica)  

classificator = lambda xy: classifier2D(xy[0],xy[1],nx,ny,b)
x,y,targets = virginica
objects = list(zip(x,y))
accuracy = calculate_accuracy(objects,targets,classificator)
precision, recall = calculate_precision_recall(objects,targets,classificator)
accuracies.append(accuracy)
precisions.append (precision)
recalls.append (recall)


accuracy = (accuracies[0]+accuracies[1]+accuracies[2])/3
precision = (precisions[0]+precisions[1]+precisions[2])/3
recall = (recalls[0]+recalls[1]+recalls[2])/3

print(f"{accuracy:.2f} | {precision:.2f} | {recall:.2f}")
ax.grid()
#plt.show()

    
    
#ax = fig.add_subplot(projection='3d')
 
 
markers_dict = {
    "Iris-setosa" : "s", 
    "Iris-versicolor" : "o",
    "Iris-virginica" : "^"   
    }

classes = data_frame["Class"].unique()

#for clss in classes:
#    plot_data = data_frame[data_frame["Class"] == clss]
 #   marker = markers_dict[clss]
 #   x = plot_data.iloc[:,0]
  #  y = plot_data.iloc[:,1]
  #  z = plot_data.iloc[:,2]
  #  ax.scatter(x,y,z)

#xx = np.linspace(0,5,100)

#X = np.arange(0, 160, 2)
#Y = np.arange(0, 160, 2)
#X, Y = np.meshgrid(X, Y)
#Z = np.sqrt(X+Y)
#ax.plot_surface(X, Y, Z, alpha=0.2)
plt.show()
