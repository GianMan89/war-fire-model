# TODO what goes in here?
# TODO prepare everything for the input to the model
# TODO fire count needs to be calculated
# TODO Time series data needs to be generated

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
    def __init__(self, fire_data, static_data, weather_data, start_date, end_date):
        """
        Initializes the feature engineering process with the provided datasets and date range.
        Args:
            fire_data (DataFrame): DataFrame containing fire-related data.
            static_data (DataFrame): DataFrame containing static data.
            weather_data (DataFrame): DataFrame containing weather-related data.
            start_date (str): The start date for the data in 'YYYY-MM-DD' format.
            end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        """
        self.fire_data = fire_data
        self.static_data = static_data
        self.weather_data = weather_data
        self.start_date = start_date
        self.end_date = end_date

    def generate_fire_time_series(self):
        """
        Generates a time series dataset for fire data within grid cells.
        This method processes fire data to create a time series for each unique grid cell. 
        It ensures that the time series is continuous from the start date to the end date, 
        filling in any missing dates with zero values. The resulting time series data 
        includes both dynamic and static features for each grid cell.
        Returns:
            pd.DataFrame: A concatenated DataFrame containing the time series data for all grid cells.
                  Each row represents a day within the specified date range for a grid cell.
                  The DataFrame includes dynamic features (e.g., 'ACQ_DATE', 'DAY_OF_YEAR', 
                  'FIRE_COUNT_CELL') and static features (all other columns in the original data 
                  except 'ACQ_DATE', 'DAY_OF_YEAR', and 'FIRE_COUNT_CELL').
        """
        time_series_data = {}
        for cell in self.fire_data['GRID_CELL'].unique():
            cell_data = self.fire_data[self.fire_data['GRID_CELL'] == cell]
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
    
    def transform(self):
        """
        Transforms the fire data by performing several operations:
        1. Sorts the fire data by acquisition date ('ACQ_DATE').
        2. Adds a new column 'FIRE_COUNT_CELL' which counts the number of fires per grid cell per acquisition date.
        3. Generates a fire time series and updates the fire data.
        4. Merges the fire data with static data based on latitude and longitude.
        5. Converts the 'ACQ_DATE' column to datetime.date format.
        6. Merges the fire data with weather data based on oblast ID and acquisition date.
        Returns:
            pd.DataFrame: The transformed fire data.
        """
        self.fire_data.sort_values('ACQ_DATE', inplace=True)
        self.fire_data['FIRE_COUNT_CELL'] = self.fire_data.groupby(['GRID_CELL', 'ACQ_DATE'])['ACQ_DATE'].transform('count')
        self.fire_data = self.generate_fire_time_series()
        self.fire_data = pd.merge(self.fire_data, self.static_data, how='left', on=['LATITUDE', 'LONGITUDE'])
        self.fire_data['ACQ_DATE'] = self.fire_data['ACQ_DATE'].apply(lambda x: pd.to_datetime(x).date())
        self.fire_data = pd.merge(self.fire_data, self.weather_data, how='left', on=['OBLAST_ID', 'ACQ_DATE'])
        return self.fire_data
    

def main():
    start_date = pd.to_datetime('2020-01-01').date()
    end_date = pd.to_datetime('2022-12-31').date()
    static_data = DataLoader.load_static_data(resolution="50km")
    fire_data, weather_data = DataLoader.load_dynamic_data(start_date=start_date, end_date=end_date)
    features = FeatureEngineering(fire_data, static_data, weather_data, start_date=start_date, end_date=end_date).transform()
    print("Shape of the features DataFrame:", features.shape)
    print(features[['ACQ_DATE', 'FIRE_COUNT_CELL']])

if __name__ == "__main__":
    main()