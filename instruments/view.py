import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import scipy.stats


def print_stats(stats: list):
    print(f"Accuracy: {stats[0]:.2f}")
    print(f"Precision: {stats[1]:.2f}")
    print(f"Recall: {stats[2]:.2f}")
    
def show_confusion_matrix(matrix: np.ndarray, labels: list) -> None:
    fig, axe = plt.subplots()
    graph = axe.matshow(matrix,cmap = "Blues")
    axe.set_xlabel("Predicted class")
    axe.set_ylabel("Observed class")
    axe.set_xticklabels(labels)
    axe.set_yticklabels(labels)
    fig.colorbar(graph, ax = axe) 