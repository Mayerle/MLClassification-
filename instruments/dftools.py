import pandas as pd
import numpy as np
import scipy.stats

class Encoder:
    __CLASS_COLUMN_NAME = "Class"
    
    def __init__(self, classes: pd.Series):
        self.classes = classes
        self.temp_classes = classes

    def one_hot_encode(self) -> None:
        unique_classes = self.classes.unique().tolist()
        classes = self.classes.map(unique_classes.index)
        classes = classes.rename(self.__CLASS_COLUMN_NAME)
        self.temp_classes = classes
        
    def label_encode(self) -> None:
        unique_classes = self.classes.unique()
        classes = pd.DataFrame()
        i = len(unique_classes) - 1
        while i >= 0:
            class_ = unique_classes[i]
            name = f"{self.__CLASS_COLUMN_NAME}{i}"
            column = self.classes.map(lambda x: 1 if x == class_ else -1) 
            classes.insert(0, name, column)
            i-=1
        self.temp_classes = classes 
        
    def get_classes(self):
        return self.temp_classes


class FeatureEditor:
    __FEATURES_COLUMN_NAME = "Features"
    
    def __init__(self, features: pd.DataFrame):
        self.features = features
    
    def convert_to_onevectors(self) -> None:
        concat_one = lambda x: np.concatenate((x,[1]))
        self.features = self.features.apply(concat_one, axis=1)
        self.features = self.features.rename(self.__FEATURES_COLUMN_NAME)
        
    def standardization_normalize(self) -> None:
        get_std = lambda x: np.std(x)
        means = self.features.apply(np.mean, axis=0)
        stds = self.features.apply(get_std, axis=0)
        self.features = self.features.sub(means,axis=1).div(stds,axis=1)

    def get_features(self) -> pd.DataFrame:
        return self.features    
        
        
class ClassificationDF:
    def __init__(self, editor: FeatureEditor, encoder: Encoder):
        self.editor = editor
        self.encoder = encoder
        
    def get(self) -> pd.DataFrame:
        frames = [self.encoder.get_classes(), self.editor.get_features()]
        return pd.concat(frames, axis=1)
    
    def get_features(self) -> pd.DataFrame:
        return self.editor.get_features()
    
    def get_classes(self) -> pd.DataFrame:
        return self.encoder.get_classes()
    