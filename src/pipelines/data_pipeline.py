import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
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
from config.config import get_parameter

class DataHandlingPipeline(BaseEstimator, TransformerMixin):
    """
    A data handling pipeline to provide input data for the model pipeline.
    
    This class handles loading, transforming, and imputing missing data for both dynamic and static datasets.
    
    Attributes
    ----------
    resolution : str
        The resolution of the data files to be loaded.
    start_date : str
        The start date for the data in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data in 'YYYY-MM-DD' format.
    calib_date : str
        The calibration date used to split the data.
    
    Methods
    -------
    fit(X, y=None)
        Placeholder for fitting the data pipeline.
    transform(X)
        Loads and processes the fire, weather, and static data.
    """
    def __init__(self, resolution='50km', start_date='2015-01-01', end_date='2022-12-31', calib_date='2021-02-23'):
        self.resolution = resolution
        self.start_date = pd.to_datetime(start_date).date()
        self.end_date = pd.to_datetime(end_date).date()
        self.calib_date = pd.to_datetime(calib_date).date()
        self.feature_engineering = FeatureEngineering(start_date=self.start_date, end_date=self.end_date)

    def fit(self, X, y=None):
        # No fitting is required for this data pipeline.
        return self

    def transform(self, X):
        static_data = DataLoader.load_static_data(resolution=self.resolution)
        fire_data, weather_data = DataLoader.load_dynamic_data(start_date=self.start_date, end_date=self.end_date, resolution=self.resolution)
        time_series_data = self.feature_engineering.transform(fire_data, static_data, weather_data)
        return time_series_data

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
    """
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)

if __name__ == "__main__":
    resolution = "50km"
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    calib_date = '2022-01-01'

    # Create the data handling pipeline
    data_pipeline = Pipeline([
        ('data_handler', DataHandlingPipeline(resolution=resolution, start_date=start_date, end_date=end_date, calib_date=calib_date)),
        ('imputer', DataImputer(strategy='mean'))
    ])

    # Process the data using the pipeline
    time_series_data = data_pipeline.fit_transform(None)

    # Split the data into training and calibration sets
    feature_engineering = FeatureEngineering(start_date=start_date, end_date=end_date)
    X_train, X_calib, y_train, y_calib, ids_train, ids_calib = feature_engineering.get_train_calibration_split(time_series_data, start_date_calib=calib_date)

    # Display the shapes of the generated datasets
    print("Shape of the training data:", X_train.shape)
    print("Shape of the calibration data:", X_calib.shape)
    print("Shape of the training target variable:", y_train.shape)
    print("Shape of the calibration target variable:", y_calib.shape)