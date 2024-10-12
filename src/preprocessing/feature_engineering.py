import os
import pandas as pd
import sys

import warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

class FeatureEngineering:
    """
    FeatureEngineering class for processing and transforming fire data.
    
    Methods
    -------
    generate_fire_time_series(fire_data, start_date, end_date)
        Generates a time series dataset for fire data within grid cells.
    transform(fire_data, static_data, weather_data, start_date, end_date)
        Transform the fire data by performing several operations.
    """
    
    def __init__(self):
        pass

    @staticmethod
    def generate_fire_time_series(fire_data, start_date, end_date):
        """
        Generates a time series dataset for fire data within grid cells.
        This method processes fire data to create a time series for each unique grid cell. 
        It ensures that the time series is continuous from the start date to the end date, 
        filling in any missing dates with zero values. The resulting time series data 
        includes both dynamic and static features for each grid cell.

        Parameters
        ----------
        fire_data : pd.DataFrame
            The fire data
        start_date : datetime.date
            The start date of the data
        end_date : datetime.date
            The end date of the data

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
            cell_data = cell_data.reindex(pd.date_range(start=start_date, end=end_date, freq='D'), fill_value=0)
            cell_data['DAY_OF_YEAR'] = cell_data.index.dayofyear
            cell_data['ACQ_DATE'] = cell_data.index
            cell_data.reset_index(drop=True, inplace=True)
            cell_data = cell_data[['ACQ_DATE'] + [col for col in cell_data.columns if col != 'ACQ_DATE']]
            for col in static_data.index:
                cell_data[col] = static_data[col]
            time_series_data[cell] = cell_data
        time_series_data = pd.concat(time_series_data.values())
        return time_series_data
    
    def transform(self, fire_data, static_data, weather_data, start_date, end_date):
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
        start_date : datetime.date
            The start date of the data.
        end_date : datetime.date
            The end date of the data.

        Returns
        -------
        X : DataFrame
            The features of the data.
        y : Series
            The target variable of the data.
        ids : DataFrame
            The fire IDs and ACQ_DATE of the data.
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
        7. Splits the data into features (X), target variable (y), and IDs.
        """
        fire_data.sort_values('ACQ_DATE', inplace=True)
        fire_data['FIRE_COUNT_CELL'] = fire_data.groupby(['GRID_CELL', 'ACQ_DATE'])['ACQ_DATE'].transform('count')
        time_series_data = self.generate_fire_time_series(fire_data, start_date, end_date)
        time_series_data = pd.merge(time_series_data, static_data, how='left', on=['LATITUDE', 'LONGITUDE'])
        time_series_data['ACQ_DATE'] = time_series_data['ACQ_DATE'].apply(lambda x: pd.to_datetime(x).date())
        time_series_data = pd.merge(time_series_data, weather_data, how='left', on=['OBLAST_ID', 'ACQ_DATE'])
        X = time_series_data.drop(columns=['FIRE_COUNT_CELL', 'OBLAST_ID', 'ACQ_DATE', 'GRID_CELL', 'LATITUDE_ORIGINAL', 
                                                                                                  'LONGITUDE_ORIGINAL']).reset_index(drop=True)
        y = time_series_data['FIRE_COUNT_CELL'].reset_index(drop=True)
        ids = time_series_data[['ACQ_DATE', 'GRID_CELL']].reset_index(drop=True)
        return X, y, ids
    

def main():
    pass

if __name__ == "__main__":
    main()