import os
import sys
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
try:
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
    sys.path.append(project_root)
except Exception as e:
    raise RuntimeError(f"Failed to set up the project root: {str(e)}")

from src.preprocessing.load_data import DataLoader
from src.preprocessing.feature_engineering import FeatureEngineering
from utils.data_utils import force_datetime

class DataPipeline:
    """
    A class to handle the data pipeline for loading, transforming, and engineering features 
    for fire data analysis.

    Parameters
    ----------
    resolution : str, optional (default='50km')
        The resolution of the static data to be loaded.
    
    Attributes
    ----------
    resolution : str
        The resolution of the static data.
    feature_engineering : FeatureEngineering
        An instance of the FeatureEngineering class for transforming data.
    static_data : any
        The static data loaded based on the specified resolution.
    
    Methods
    -------
    fit()
    transform(start_date='2015-01-01', end_date='2022-02-23')
    fit_transform(start_date='2015-01-01', end_date='2022-02-23')
    """
    
    def __init__(self, resolution='50km'):
        self.resolution = resolution
        self.feature_engineering = FeatureEngineering()
        self.static_data = None

    def fit(self):
        """
        Fit the data pipeline by loading static data.
        This method attempts to load static data based on the specified resolution.
        If the data loading fails, it raises a RuntimeError with the error message.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        
        Raises
        ------
        RuntimeError
            If loading static data fails.
        """

        try:
            # Load static data based on the specified resolution
            self.static_data = DataLoader.load_static_data(resolution=self.resolution)
        except Exception as e:
            raise RuntimeError(f"Failed to load static data: {str(e)}")
        return self

    def transform(self, start_date='2015-01-01', end_date='2022-02-23'):
        """
        Transforms the data within the specified date range by loading dynamic data,
        applying feature engineering, and returning the transformed features and labels.
        
        Parameters
        ----------
        start_date : str, optional
            The start date for the data transformation in 'YYYY-MM-DD' format. Default is '2015-01-01'.
        end_date : str, optional
            The end date for the data transformation in 'YYYY-MM-DD' format. Default is '2022-02-23'.
        
        Returns
        -------
        X : array-like
            The transformed feature matrix.
        y : array-like
            The target labels.
        ids : array-like
            The identifiers for the data points.
        fire_data : DataFrame
            The raw fire data loaded for the given date range.
        
        Raises
        ------
        RuntimeError
            If there is an error during the data transformation process.
        """

        try:
            start_date, end_date = force_datetime(start_date), force_datetime(end_date)
            # Load dynamic data for the given date range
            fire_data, weather_data = DataLoader.load_dynamic_data(start_date=start_date, end_date=end_date, resolution=self.resolution)
            # Apply feature engineering to transform the data
            X, y, ids = self.feature_engineering.transform(fire_data, self.static_data, weather_data, start_date=start_date, 
                                                           end_date=end_date)
        except Exception as e:
            raise RuntimeError(f"Failed to transform data: {str(e)}")
        return X, y, ids, fire_data
    
    def fit_transform(self, start_date='2015-01-01', end_date='2022-02-23'):
        """
        Fit the model and then transform the data within the specified date range.
        
        Parameters
        ----------
        start_date : str, optional
            The start date for the data transformation in 'YYYY-MM-DD' format. Default is '2015-01-01'.
        end_date : str, optional
            The end date for the data transformation in 'YYYY-MM-DD' format. Default is '2022-02-23'.
        
        Returns
        -------
        transformed_data : any
            The transformed data within the specified date range.
        """

        self.fit()
        return self.transform(start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    try:
        resolution = "50km"
        start_date = '2020-01-01'
        end_date = '2022-12-31'
        
        # Initialize and fit-transform the data pipeline
        pipeline = DataPipeline(resolution=resolution)
        X, y, ids, fires = pipeline.fit_transform(start_date=start_date, end_date=end_date)
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("ids shape:", ids.shape)
        print("fires shape:", fires.shape)
        
        # Transform data with the loaded pipeline
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        X, y, ids, fires = pipeline.transform(start_date=start_date, end_date=end_date)
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("ids shape:", ids.shape)
        print("fires shape:", fires.shape)
    except Exception as e:
        print(f"An error occurred: {str(e)}")