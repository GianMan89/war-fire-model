import os
import pandas as pd
import sys

import warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

from src.preprocessing.load_data import DataLoader

class FeatureEngineering:
    """
    FeatureEngineering class for processing and transforming fire data.
    
    Attributes
    ----------
    start_date : str
        The start date for the data in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data in 'YYYY-MM-DD' format.
    Methods
    -------
    generate_fire_time_series(fire_data)
    transform(fire_data, static_data, weather_data)
        Transforms the fire data by performing several operations including sorting, counting, merging, and converting dates.
    get_train_calibration_split(time_series_data, start_date_calib)
        Splits the transformed data into training and calibration sets based on the specified date range.
    """
    
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def generate_fire_time_series(self, fire_data):
        """
        Generates a time series dataset for fire data within grid cells.
        This method processes fire data to create a time series for each unique grid cell. 
        It ensures that the time series is continuous from the start date to the end date, 
        filling in any missing dates with zero values. The resulting time series data 
        includes both dynamic and static features for each grid cell.

        Parameters
        ----------
        fire_data : pd.DataFrame
            The fire data with columns 'ACQ_DATE', 'DAY_OF_YEAR', 
        Returns
        -------
        pd.DataFrame
            A concatenated DataFrame containing the time series data for all grid cells.
        """
        time_series_data = {}
        for cell in fire_data['GRID_CELL'].unique():
            cell_data = fire_data[fire_data['GRID_CELL'] == cell].drop(columns=['FIRE_ID'])
            static_data = cell_data.iloc[0].drop(['ACQ_DATE', 'DAY_OF_YEAR', 'FIRE_COUNT_CELL'])
            cell_data.set_index('ACQ_DATE', inplace=True)
            cell_data.index = pd.to_datetime(cell_data.index)
            cell_data = cell_data[~cell_data.index.duplicated(keep='first')]
            cell_data = cell_data.reindex(pd.date_range(start=self.start_date, end=self.end_date, freq='D'), fill_value=0)
            cell_data['DAY_OF_YEAR'] = cell_data.index.dayofyear
            cell_data['ACQ_DATE'] = cell_data.index
            cell_data.reset_index(drop=True, inplace=True)
            cell_data = cell_data[['ACQ_DATE'] + [col for col in cell_data.columns if col != 'ACQ_DATE']]
            for col in static_data.index:
                cell_data[col] = static_data[col]
            time_series_data[cell] = cell_data
        time_series_data = pd.concat(time_series_data.values())
        return time_series_data
    
    def transform(self, fire_data, static_data, weather_data):
        """
        Transform the fire data by performing several operations.

        Parameters
        ----------
        fire_data : pd.DataFrame
            The raw fire data.
        static_data : pd.DataFrame
            The static data.
        weather_data : pd.DataFrame
            The weather data.
        Returns
        -------
        pd.DataFrame
            The transformed fire data.
        Notes
        -----
        The transformation includes the following steps:
        Transforms the fire data by performing several operations:
        1. Sorts the fire data by acquisition date ('ACQ_DATE').
        2. Adds a new column 'FIRE_COUNT_CELL' which counts the number of fires per grid cell per acquisition date.
        3. Generates a fire time series and updates the fire data.
        4. Merges the fire data with static data based on latitude and longitude.
        5. Converts the 'ACQ_DATE' column to datetime.date format.
        6. Merges the fire data with weather data based on oblast ID and acquisition date.
        7. Checks for NaN values and replaces them with the closest non-NaN value based on latitude, longitude, and day of year.
        """
        fire_data.sort_values('ACQ_DATE', inplace=True)
        fire_data['FIRE_COUNT_CELL'] = fire_data.groupby(['GRID_CELL', 'ACQ_DATE'])['ACQ_DATE'].transform('count')
        time_series_data = self.generate_fire_time_series(fire_data)
        time_series_data = pd.merge(time_series_data, static_data, how='left', on=['LATITUDE', 'LONGITUDE'])
        time_series_data['ACQ_DATE'] = time_series_data['ACQ_DATE'].apply(lambda x: pd.to_datetime(x).date())
        time_series_data = pd.merge(time_series_data, weather_data, how='left', on=['OBLAST_ID', 'ACQ_DATE'])
        if time_series_data.isna().sum().sum() > 0:
            lat_min, lat_max = time_series_data['LATITUDE'].min(), time_series_data['LATITUDE'].max()
            lon_min, lon_max = time_series_data['LONGITUDE'].min(), time_series_data['LONGITUDE'].max()
            doy_min, doy_max = time_series_data['DAY_OF_YEAR'].min(), time_series_data['DAY_OF_YEAR'].max()
            nan_rows = time_series_data[time_series_data.isna().any(axis=1)]
            non_nan_rows = time_series_data[time_series_data.notna().all(axis=1)]
            for idx, nan_row in nan_rows.iterrows(): 
                    filtered_rows = non_nan_rows[
                        (non_nan_rows['DAY_OF_YEAR'] >= nan_row['DAY_OF_YEAR'] - 7) &
                        (non_nan_rows['DAY_OF_YEAR'] <= nan_row['DAY_OF_YEAR'] + 7) &
                        (non_nan_rows['LATITUDE'] >= nan_row['LATITUDE'] - 1.0) &
                        (non_nan_rows['LATITUDE'] <= nan_row['LATITUDE'] + 1.0) &
                        (non_nan_rows['LONGITUDE'] >= nan_row['LONGITUDE'] - 1.0) &
                        (non_nan_rows['LONGITUDE'] <= nan_row['LONGITUDE'] + 1.0)
                    ]
                    distances = filtered_rows.apply(
                        lambda row: (
                            ((row['LATITUDE'] - nan_row['LATITUDE']) / (lat_max - lat_min))**2 + 
                            ((row['LONGITUDE'] - nan_row['LONGITUDE']) / (lon_max - lon_min))**2 + 
                            ((row['DAY_OF_YEAR'] - nan_row['DAY_OF_YEAR']) / (doy_max - doy_min))**2
                        )**0.5, 
                        axis=1
                    )
                    closest_row_idx = distances.idxmin()
                    for col in nan_row.index:
                        if pd.isna(nan_row[col]):
                            time_series_data.at[idx, col] = time_series_data.at[closest_row_idx, col]
        return time_series_data
    
    def get_train_calibration_split(self, time_series_data, start_date_calib):
        """
        Splits the transformed data into training and calibration sets.
        This method splits the transformed data into training and calibration sets based on the specified date range.
        The training set includes data from the start date to the day before the start_date_calib date, 
        while the calibration set includes data from the start_date_calib date to the end date.

        Parameters
        ----------
        time_series_data : DataFrame
            The input time series data containing features and target variables.
        start_date_calib : datetime
            The start date for the calibration set.
        Returns
        -------
        X_train : DataFrame
            The features of the training data.
        X_calib : DataFrame
            The features of the calibration data.
        y_train : Series
            The target variable of the training data.
        y_calib : Series
            The target variable of the calibration data.
        ids_train : DataFrame
            The fire IDs and ACQ_DATE of the training data.
        ids_calib : DataFrame
            The fire IDs and ACQ_DATE of the calibration data.
        """
        X_train = time_series_data[time_series_data['ACQ_DATE'] < start_date_calib].drop(columns=['FIRE_COUNT_CELL', 'OBLAST_ID',
                                                                                                  'ACQ_DATE', 'GRID_CELL', 'LATITUDE_ORIGINAL', 
                                                                                                  'LONGITUDE_ORIGINAL']).reset_index(drop=True)
        X_calib = time_series_data[time_series_data['ACQ_DATE'] >= start_date_calib].drop(columns=['FIRE_COUNT_CELL', 'OBLAST_ID',
                                                                                                   'ACQ_DATE', 'GRID_CELL', 'LATITUDE_ORIGINAL', 
                                                                                                  'LONGITUDE_ORIGINAL']).reset_index(drop=True)
        y_train = time_series_data[time_series_data['ACQ_DATE'] < start_date_calib]['FIRE_COUNT_CELL'].reset_index(drop=True)
        y_calib = time_series_data[time_series_data['ACQ_DATE'] >= start_date_calib]['FIRE_COUNT_CELL'].reset_index(drop=True)
        # Get the fire IDs and acquisition dates for the training and calibration sets but only for those dates
        ids_train = time_series_data[time_series_data['ACQ_DATE'] < start_date_calib][['ACQ_DATE', 'GRID_CELL']].reset_index(drop=True)
        ids_calib = time_series_data[time_series_data['ACQ_DATE'] >= start_date_calib][['ACQ_DATE', 'GRID_CELL']].reset_index(drop=True)
        return X_train, X_calib, y_train, y_calib, ids_train, ids_calib

    def get_test_data(self, time_series_data):
        """
        Splits the transformed data into training and calibration sets.
        This method splits the transformed data into training and calibration sets based on the specified date range.
        The training set includes data from the start date to the day before the start_date_calib date, 
        while the calibration set includes data from the start_date_calib date to the end date.

        Parameters
        ----------
        time_series_data : DataFrame
            The input time series data containing features and target variables.
        start_date_calib : datetime
            The start date for the calibration set.
        Returns
        -------
        X_train : DataFrame
            The features of the training data.
        X_calib : DataFrame
            The features of the calibration data.
        y_train : Series
            The target variable of the training data.
        y_calib : Series
            The target variable of the calibration data.
        ids_train : DataFrame
            The fire IDs and ACQ_DATE of the training data.
        ids_calib : DataFrame
            The fire IDs and ACQ_DATE of the calibration data.
        """
        X_test = time_series_data.drop(columns=['FIRE_COUNT_CELL', 'OBLAST_ID', 'ACQ_DATE', 'GRID_CELL', 'LATITUDE_ORIGINAL', 
                                                                                                  'LONGITUDE_ORIGINAL']).reset_index(drop=True)
        y_test = time_series_data['FIRE_COUNT_CELL'].reset_index(drop=True)
        ids_test = time_series_data[['ACQ_DATE', 'GRID_CELL']].reset_index(drop=True)
        return X_test, y_test, ids_test
    

def main():
    resolution = "50km"
    start_date = pd.to_datetime('2020-01-01').date()
    end_date = pd.to_datetime('2022-12-31').date()
    calib_date = pd.to_datetime('2022-01-01').date()
    static_data = DataLoader.load_static_data(resolution=resolution)
    fire_data, weather_data = DataLoader.load_dynamic_data(start_date=start_date, end_date=end_date, resolution=resolution)
    feature_engineering = FeatureEngineering(start_date=start_date, end_date=end_date)
    time_series_data = feature_engineering.transform(fire_data, static_data, weather_data)
    X_train, X_calib, y_train, y_calib, ids_train, ids_calib = feature_engineering.get_train_calibration_split(time_series_data, 
                                                                                                               start_date_calib=calib_date)
    print("Shape of the training data:", X_train.shape)
    print("Shape of the calibration data:", X_calib.shape)
    print("Shape of the training target variable:", y_train.shape)
    print("Shape of the calibration target variable:", y_calib.shape)
    print("Shape of the training IDs:", ids_train.shape)
    print("Shape of the calibration IDs:", ids_calib.shape)

if __name__ == "__main__":
    main()