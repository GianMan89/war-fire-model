import os
import geopandas as gpd
import pandas as pd
import sys

import warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

from utils.file_utils import get_path
from config.config import get_parameter
from utils.data_utils import round_lat_lon

class DataLoader:
    """
    DataLoader is a utility class for loading and preprocessing fire and weather data.

    Methods
    -------
    filter_date(data, start_date, end_date)
    preprocess_fire_data(data)
    filter_border(data)
    load_dynamic_data(start_date, end_date)
    load_static_data(resolution="50km")
    """
    @staticmethod
    def filter_date(data, start_date, end_date):
        """
        Filters the data based on a date range.

        Parameters
        ----------
        data : DataFrame
            Fire data.
        start_date : datetime.date
            Start date for filtering.
        end_date : datetime.date
            End date for filtering.
        Returns
        -------
        DataFrame
            Filtered fire data.
        """
        data['ACQ_DATE'] = pd.to_datetime(data['ACQ_DATE']).dt.date
        filtered_data = data[(data['ACQ_DATE'] >= start_date) & (data['ACQ_DATE'] <= end_date)]
        filtered_data.reset_index(drop=True, inplace=True)
        return filtered_data
    
    @staticmethod
    def preprocess_fire_data(data):
        """
        Preprocesses the fire data by dropping unnecessary columns and generating additional features.

        Parameters
        ----------
        data : DataFrame
            Raw fire data.
        Returns
        -------
        DataFrame
            Preprocessed fire data.
        """
        data['ACQ_DATE'] = data['ACQ_DATE'].apply(lambda x: pd.to_datetime(x).date())
        data['DAY_OF_YEAR'] = data['ACQ_DATE'].apply(lambda x: x.timetuple().tm_yday)
        data['FIRE_ID'] = data.index
        return data
    
    @staticmethod
    def filter_border(data):
        """
        Filters the data based on administrative borders.

        Parameters
        ----------
        data : DataFrame
            Fire data.
        Returns
        -------
        DataFrame
            Filtered fire data.
        """
        borders = gpd.read_file(get_path("border_data_dir"))
        data_gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['LONGITUDE'], data['LATITUDE']))
        data_gdf.set_crs(epsg=get_parameter("border_epsg"), inplace=True)
        borders.set_crs(epsg=get_parameter("border_epsg"), inplace=True)
        data_inside = gpd.sjoin(data_gdf, borders, how='inner')
        data_inside.drop(columns=['geometry', 'source', 'name', 'index_right'], inplace=True)
        data_inside.reset_index(drop=True, inplace=True)
        data_inside.columns = map(str.upper, data_inside.columns)
        data_inside.drop(columns=['ID'], inplace=True)
        data_inside.rename(columns={'LATITUDE': 'LATITUDE_ORIGINAL', 'LONGITUDE': 'LONGITUDE_ORIGINAL'}, inplace=True)
        # Round the latitude and longitude to the nearest .5 or .0
        data_inside['LATITUDE'], data_inside['LONGITUDE'] = zip(*data_inside.apply(lambda x: round_lat_lon(x['LATITUDE_ORIGINAL'], x['LONGITUDE_ORIGINAL']), axis=1))
        data_inside['GRID_CELL'] = data_inside['LATITUDE'].astype(str) + '_' + data_inside['LONGITUDE'].astype(str)
        data_inside = data_inside[['FIRE_ID', 'LATITUDE_ORIGINAL', 'LONGITUDE_ORIGINAL', 'LATITUDE', 'LONGITUDE', 'GRID_CELL', 'ACQ_DATE', 'DAY_OF_YEAR']]
        return data_inside

    @staticmethod
    def load_dynamic_data(start_date, end_date):
        """
        Loads and processes dynamic fire and weather data within a specified date range.
        This function reads CSV files from specified directories, filters them by the given date range,
        and returns concatenated and deduplicated dataframes for fire and weather data.

        Parameters
        ----------
        start_date : datetime
            The start date for filtering the data.
        end_date : datetime
            The end date for filtering the data.
        Returns
        -------
        tuple
            A tuple containing two pandas DataFrames:
            - weather_data (DataFrame): The concatenated and deduplicated weather data within the date range.
        """
        fire_data, weather_data = [], []
        folders = [get_path("weather_data_dir"), get_path("fire_data_dir")]
        for folder_path in folders:
            for file_path in os.listdir(folder_path):
                if not file_path.endswith('.csv'):
                    continue
                year = int(file_path.split('_')[-1][:4])
                if start_date.year <= year <= end_date.year:
                    df = pd.read_csv(f"{folder_path}/{file_path}")
                    if 'fire' in file_path:
                        fire_data.append(df)
                    elif 'weather' in file_path:
                        weather_data.append(df)
        fire_data = pd.concat(fire_data).drop_duplicates()
        weather_data = pd.concat(weather_data).drop_duplicates()
        
        fire_data = DataLoader.filter_date(fire_data, start_date, end_date)
        weather_data = DataLoader.filter_date(weather_data, start_date, end_date)

        fire_data = DataLoader.preprocess_fire_data(fire_data)
        fire_data = DataLoader.filter_border(fire_data)
        
        return fire_data, weather_data
    
    @staticmethod
    def load_static_data(resolution="50km"):
        """
        Loads and merges static data from multiple directories based on the specified resolution.
        This function reads CSV files from predefined directories, filters them by the given resolution,
        and merges them into a single DataFrame. The merging is done on the 'LONGITUDE' and 'LATITUDE' columns.

        Parameters
        ----------
        resolution : str, optional
            The resolution of the data files to be loaded. Default is "50km".
        Returns
        -------
        DataFrame
            A DataFrame containing the merged data from all the relevant CSV files.
        """
        dataframes = []
        folders = [get_path("oblasts_data_dir"), get_path("land_use_data_dir"), get_path("pop_density_data_dir")]
        for folder_path in folders:
            for file_path in os.listdir(folder_path):
                if file_path.endswith(f'{resolution}.csv'):
                    df = pd.read_csv(f"{folder_path}/{file_path}")
                else:
                    continue
                df.columns = map(str.upper, df.columns)
                dataframes.append(df)
        if dataframes:
            merged_data = dataframes[0]
            for df in dataframes[1:]:
                merged_data = pd.merge(merged_data, df, on=['LONGITUDE', 'LATITUDE'], how='outer')
            return merged_data
        return merged_data

def main():
    static_data = DataLoader.load_static_data(resolution="50km")
    fire_data, weather_data = DataLoader.load_dynamic_data(start_date=pd.to_datetime('2020-01-01').date(), end_date=pd.to_datetime('2022-12-31').date())
    print("Fire Data - Min Date:", fire_data['ACQ_DATE'].min(), "Max Date:", fire_data['ACQ_DATE'].max())
    print("Weather Data - Min Date:", weather_data['ACQ_DATE'].min(), "Max Date:", weather_data['ACQ_DATE'].max())
    print("Static Data - Shape:", static_data.shape)
    print("Fire Data - Shape:", fire_data.shape)
    print("Weather Data - Shape:", weather_data.shape)

if __name__ == "__main__":
    main()