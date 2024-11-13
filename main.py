import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import scipy.stats
from instruments.stats import *
from instruments.dftools import *
from instruments.functions import *
from instruments.view import *
RANDOM_STATE = 3


df = pd.read_csv("datasets/iris.csv")
class_names = ["","Setosa","Versicolor","Virginica"]

editor = FeatureEditor(df.iloc[:,1:-1])
encoder = Encoder(df.iloc[:,-1])
cdf = ClassificationDF(editor, encoder)

encoder.label_encode()
editor.standardization_normalize()
editor.convert_to_onevectors()
print(cdf.get())



