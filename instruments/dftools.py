import pandas as pd
import numpy as np
import scipy.stats
import math

class Encoder:
    CLASS_COLUMN_NAME = "Class"
    
    def __init__(self, classes: pd.Series):
        self.classes = classes
        self.temp_classes = classes
        __is_label_encode = False
        
    def one_hot_encode(self) -> None:
        unique_classes = self.classes.unique().tolist()
        classes = self.classes.map(unique_classes.index)
        classes = classes.rename(self.CLASS_COLUMN_NAME)
        self.temp_classes = classes
        self.__is_label_encode = False
        
    def label_encode(self) -> None:
        unique_classes = self.classes.unique()
        classes = pd.DataFrame()
        i = len(unique_classes) - 1
        while i >= 0:
            class_ = unique_classes[i]
            name = f"{self.CLASS_COLUMN_NAME}{i}"
            column = self.classes.map(lambda x: 1 if x == class_ else -1) 
            classes.insert(0, name, column)
            i-=1
        self.temp_classes = classes 
        self. __is_label_encode = True
        
    def get_classes(self, index: int = -1):
        if(index == -1):
            return self.temp_classes
        if(self.__is_label_encode):
            return self.temp_classes[f"{self.CLASS_COLUMN_NAME}{index}"]
        else:
            raise Exception("Can not get class using one hot encode!")


class FeatureEditor:
    FEATURES_COLUMN_NAME = "Features"
    
    def __init__(self, features: pd.DataFrame):
        self.features = features
        self.features_length = features.shape[1]
    
    def convert_to_onevectors(self) -> None:
        concat_one = lambda x: np.concatenate((x,[1]))
        self.features = self.features.apply(concat_one, axis=1)
        self.features = self.features.rename(self.FEATURES_COLUMN_NAME)
        self.features_length += 1
        
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
        df = pd.concat(frames, axis=1)
        return df
    
    def get_features(self) -> pd.DataFrame:
        return self.editor.get_features()
    
    def get_classes(self, index: int = -1) -> pd.DataFrame:
        return self.encoder.get_classes(index)
    def get_features_length(self) -> int:
            return self.editor.features_length
    
    
DEFAULT_SEED = 3
def sample_df(df: pd.DataFrame, seed: float = DEFAULT_SEED, train_volume: float = 0.7, validate_volume: float = 0) -> list:
    df = df.sample(frac=1,random_state=seed)
    
    train_n = math.trunc(train_volume*df.shape[0])
    train   = df.iloc[:train_n,:]
    test    = df.iloc[train_n:,:]
    
    validate_n = math.trunc((1-validate_volume)*train.shape[0])
    validate   = train.iloc[validate_n:,:]
    train      = train.iloc[:validate_n,:]
    return [train, validate, test]