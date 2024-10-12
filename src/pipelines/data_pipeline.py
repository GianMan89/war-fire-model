import os
import sys
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

from src.preprocessing.load_data import DataLoader
from src.preprocessing.feature_engineering import FeatureEngineering

class DataHandler(BaseEstimator, TransformerMixin):
    """"
    DataHandler is a custom transformer for handling and processing data for the war-fire model.
    
    Parameters
    ----------
    resolution : str, default='50km'
        The spatial resolution for loading static and dynamic data.
    
    Attributes
    ----------
    resolution : str
        The spatial resolution for loading static and dynamic data.
    feature_engineering : FeatureEngineering
        An instance of the FeatureEngineering class for transforming the data.
    static_data : Any
        The static data loaded based on the specified resolution.
    
    Methods
    -------
    fit(X=None, y=None)
        Loads static data based on the specified resolution.
    transform(X=None, y=None, start_date='2015-01-01', end_date='2022-02-23')
        Loads dynamic data for the given date range and applies feature engineering to transform the data.
    fit_transform(X=None, y=None, start_date='2015-01-01', end_date='2022-02-23')
        Fits the transformer to the data and transforms the data.
    """
    def __init__(self, resolution='50km'):
        self.resolution = resolution
        self.feature_engineering = FeatureEngineering()
        self.static_data = None

    def fit(self, X=None, y=None):
        self.static_data = DataLoader.load_static_data(resolution=self.resolution)
        return self

    def transform(self, X=None, y=None, start_date='2015-01-01', end_date='2022-02-23'):
        fire_data, weather_data = DataLoader.load_dynamic_data(start_date=start_date, end_date=end_date, resolution=self.resolution)
        X, y, ids = self.feature_engineering.transform(fire_data, self.static_data, weather_data, start_date=start_date, 
                                                       end_date=end_date)
        return X, y, ids
    
    def fit_transform(self, X=None, y=None, start_date='2015-01-01', end_date='2022-02-23'):
        self.fit(X, y)
        return self.transform(X, y, start_date=start_date, end_date=end_date)

class DataImputer(BaseEstimator, TransformerMixin):
    """
    DataImputer class to handle imputation of missing data.
    
    Attributes
    ----------
    strategy : str
        The imputation strategy. Default is 'mean'.
    
    Methods
    -------
    fit(X, y=None)
        Fits the imputer to the data.
    transform(X)
        Imputes missing values in the data.
    fit_transform(X, y=None)
        Fits the imputer to the data and imputes missing values.
    """
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
class DataPipeline:
    """
    DataPipeline class to handle the entire data processing pipeline.
    
    Attributes
    ----------
    resolution : str
        The resolution of the data files to be loaded.
    
    Methods
    -------
    fit(start_date='2015-01-01', end_date='2022-02-23')
        Fits the data handler and imputer to the data.
    transform(start_date='2015-01-01', end_date='2022-02-23')
        Transforms the data using the data handler and imputer.
    fit_transform(start_date='2015-01-01', end_date='2022-02-23')
        Fits the data handler and imputer to the data and transforms the data.
    """
    def __init__(self, resolution='50km'):
        self.resolution = resolution
        self.data_handler = DataHandler(resolution=resolution)
        self.imputer = DataImputer(strategy='mean')

    def fit(self, start_date='2015-01-01', end_date='2022-02-23'):
        start_date, end_date = self.force_datetime(start_date), self.force_datetime(end_date)
        data, _, _ = self.data_handler.fit_transform(X=None, y=None, start_date=start_date, end_date=end_date)
        self.imputer.fit(data)
        return self

    def transform(self, start_date='2015-01-01', end_date='2022-02-23'):
        start_date, end_date = self.force_datetime(start_date), self.force_datetime(end_date)
        X, y, ids = self.data_handler.transform(X=None, y=None, start_date=start_date, end_date=end_date)
        X_imputed = self.imputer.transform(X)
        return X_imputed, y, ids
    
    def fit_transform(self, start_date='2015-01-01', end_date='2022-02-23'):
        start_date, end_date = self.force_datetime(start_date), self.force_datetime(end_date)
        self.fit(start_date=start_date, end_date=end_date)
        return self.transform(start_date=start_date, end_date=end_date)
    
    @staticmethod
    def force_datetime(date):
        return pd.to_datetime(date).date()

if __name__ == "__main__":
    resolution = "50km"
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    
    pipeline = DataPipeline(resolution=resolution)
    X, y, ids = pipeline.fit_transform(start_date=start_date, end_date=end_date)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("ids shape:", ids.shape)
    
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    X, y, ids = pipeline.transform(start_date=start_date, end_date=end_date)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("ids shape:", ids.shape)