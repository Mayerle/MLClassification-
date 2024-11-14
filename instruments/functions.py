import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.stats


def sigmoid(margin):
    return (1+np.exp(-margin))**-1

def quadric(margin):
    return (1-margin)**2




def l1_norm(vector) -> float:
    sum = 0
    for x in vector:
        sum += abs(x)
    return float(sum)

def l2_norm(vector) -> float:
    sum = 0
    for x in vector:
        sum += x**2
    return float(np.sqrt(sum))


def logistic_loss(normal: list, objects: list, targets: list) -> float:
    summ = 0
    for feature, target in zip(objects, targets):
        margin = target*np.dot(feature, normal)
        summ -= np.log(sigmoid(margin))
    return summ
     
def mse_biclass_loss(normal: list, objects: list, targets: list) -> float:
    summ = 0
    for feature, target in zip(objects, targets):
        margin = target*np.dot(feature, normal)
        summ += quadric(margin)
    return summ

