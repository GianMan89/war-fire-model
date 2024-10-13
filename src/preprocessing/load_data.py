import os
import geopandas as gpd
import pandas as pd
import sys
import warnings

warnings.filterwarnings("ignore")

# Add the project root to the Python path
try:
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
    sys.path.append(project_root)
except Exception as e:
    raise RuntimeError(f"Failed to add project root to Python path: {e}")

try:
    from utils.file_utils import get_path
    from config.config import get_parameter
    from utils.data_utils import round_lat_lon, force_datetime
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    sys.exit(1)

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

        Raises
        ------
        KeyError
            If the 'ACQ_DATE' column is missing in the data.
        Exception
            If an error occurs while filtering the data by date.
        """
        try:
            data['ACQ_DATE'] = pd.to_datetime(data['ACQ_DATE']).dt.date
            filtered_data = data[(data['ACQ_DATE'] >= start_date) & (data['ACQ_DATE'] <= end_date)]
            filtered_data.reset_index(drop=True, inplace=True)
            return filtered_data
        except KeyError as e:
            raise KeyError(f"Missing 'ACQ_DATE' column in data: {e}") from e
        except Exception as e:
            raise Exception(f"Error occurred while filtering data by date: {e}") from e
    
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

        Raises
        ------
        KeyError
            If the required columns are missing in the data.
        Exception
            If an error occurs during preprocessing the fire
        """
        try:
            data['ACQ_DATE'] = data['ACQ_DATE'].apply(lambda x: pd.to_datetime(x).date())
            data['DAY_OF_YEAR'] = data['ACQ_DATE'].apply(lambda x: x.timetuple().tm_yday)
            data['FIRE_ID'] = data.index
            return data
        except KeyError as e:
            raise KeyError(f"Missing required columns in data: {e}") from e
        except Exception as e:
            raise Exception(f"Error occurred during preprocessing fire data: {e}") from e
    
    @staticmethod
    def filter_border(data, resolution="50km"):
        """
        Filters the data based on administrative borders.

        Parameters
        ----------
        data : DataFrame
            Fire data.
        resolution : str, optional
            The resolution of the data files to be loaded. Default is "50km".
        
        Returns
        -------
        DataFrame
            Filtered fire data.

        Raises
        ------
        FileNotFoundError
            If the border data file is not found.
        KeyError
            If the required columns are missing in the data for spatial join.
        Exception
            If an error occurs while filtering the data inside the borders.
        """
        try:
            borders = gpd.read_file(get_path("border_data_dir"))
            data_gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['LONGITUDE'], data['LATITUDE']))
            data_gdf.set_crs(epsg=get_parameter("border_epsg"), inplace=True)
            borders.set_crs(epsg=get_parameter("border_epsg"), inplace=True)

            # Perform spatial join to filter data inside the borders
            data_inside = gpd.sjoin(data_gdf, borders, how='inner')

            # Drop unnecessary columns
            data_inside.drop(columns=['geometry', 'source', 'name', 'index_right'], inplace=True)
            data_inside.reset_index(drop=True, inplace=True)

            # Rename and create new columns
            data_inside.columns = map(str.upper, data_inside.columns)
            data_inside.drop(columns=['ID'], inplace=True, errors='ignore')
            data_inside.rename(columns={'LATITUDE': 'LATITUDE_ORIGINAL', 'LONGITUDE': 'LONGITUDE_ORIGINAL'}, inplace=True)

            # Round latitude and longitude based on resolution
            data_inside['LATITUDE'], data_inside['LONGITUDE'] = zip(*data_inside.apply(lambda x: round_lat_lon(x['LATITUDE_ORIGINAL'], 
                                                                                                               x['LONGITUDE_ORIGINAL'], 
                                                                                                               resolution=resolution), 
                                                                                                               axis=1))

            # Create GRID_CELL column
            data_inside['GRID_CELL'] = data_inside['LATITUDE'].astype(str) + '_' + data_inside['LONGITUDE'].astype(str)

            # Select relevant columns
            data_inside = data_inside[['FIRE_ID', 'LATITUDE_ORIGINAL', 'LONGITUDE_ORIGINAL', 'LATITUDE', 'LONGITUDE', 
                                       'GRID_CELL', 'ACQ_DATE', 'DAY_OF_YEAR']]
            return data_inside
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Border data file not found: {e}") from e
        except KeyError as e:
            raise KeyError(f"Missing required columns in data for spatial join: {e}") from e
        except Exception as e:
            raise Exception(f"Error occurred while filtering borders: {e}") from e

    @staticmethod
    def load_dynamic_data(start_date, end_date, resolution="50km"):
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
        resolution : str, optional
            The resolution of the data files to be loaded. Default is "50km".
        
        Returns
        -------
        tuple
            A tuple containing two pandas DataFrames:
            - fire_data (DataFrame): The processed fire data within the date range.
            - weather_data (DataFrame): The concatenated and deduplicated weather data within the date range.

        Raises
        ------
        FileNotFoundError
            If no fire or weather data files are found for the specified date range.
        ValueError
            If one of the CSV files is empty.
        Exception
            If an error occurs while loading dynamic data.
        """
        # Convert start and end dates to datetime.date
        start_date, end_date = force_datetime(start_date), force_datetime(end_date)
        fire_data, weather_data = [], []
        try:
            # Iterate over weather and fire data directories
            folders = [get_path("weather_data_dir"), get_path("fire_data_dir")]
            for folder_path in folders:
                for file_path in os.listdir(folder_path):
                    if not file_path.endswith('.csv'):
                        continue
                    year = int(file_path.split('_')[-1][:4])
                    # Check if the file falls within the specified date range
                    if start_date.year <= year <= end_date.year:
                        df = pd.read_csv(f"{folder_path}/{file_path}")
                        if 'fire' in file_path:
                            fire_data.append(df)
                        elif 'weather' in file_path:
                            weather_data.append(df)

            if not fire_data or not weather_data:
                raise FileNotFoundError("No fire or weather data files found for the specified date range.")
            
            # Concatenate and drop duplicates
            fire_data = pd.concat(fire_data).drop_duplicates()
            weather_data = pd.concat(weather_data).drop_duplicates()
            
            # Filter and preprocess data
            fire_data = DataLoader.filter_date(fire_data, start_date, end_date)
            weather_data = DataLoader.filter_date(weather_data, start_date, end_date)
            fire_data = DataLoader.preprocess_fire_data(fire_data)
            fire_data = DataLoader.filter_border(fire_data, resolution=resolution)
            
            return fire_data, weather_data
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error while loading data files: {e}") from e
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"One of the CSV files is empty: {e}") from e
        except Exception as e:
            raise Exception(f"Error occurred while loading dynamic data: {e}") from e
    
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
            Else, returns None if no data is found for the specified resolution.

        Raises
        ------
        FileNotFoundError
            If no static data files are found for the specified resolution.
        ValueError
            If one of the CSV files is empty.
        Exception
            If an error occurs while loading static data.
        """
        dataframes = []
        try:
            # Iterate over static data directories
            folders = [get_path("oblasts_data_dir"), get_path("land_use_data_dir"), get_path("pop_density_data_dir")]
            for folder_path in folders:
                for file_path in os.listdir(folder_path):
                    # Check if file matches the specified resolution
                    if file_path.endswith(f'{resolution}.csv'):
                        df = pd.read_csv(f"{folder_path}/{file_path}")
                    else:
                        continue
                    df.columns = map(str.upper, df.columns)
                    dataframes.append(df)

            if not dataframes:
                raise FileNotFoundError("No static data files found for the specified resolution.")

            # Merge dataframes on LATITUDE and LONGITUDE
            merged_data = dataframes[0]
            for df in dataframes[1:]:
                merged_data = pd.merge(merged_data, df, on=['LONGITUDE', 'LATITUDE'], how='outer')

            # Reduce the impact of outliers by capping the population density at the 95th percentile
            if 'POP_DENSITY' in merged_data.columns:
                percentile_95_pop_density = merged_data['POP_DENSITY'].quantile(0.95)
                merged_data.loc[merged_data['POP_DENSITY'] > percentile_95_pop_density, 'POP_DENSITY'] = percentile_95_pop_density

            return merged_data
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error while loading static data files: {e}") from e
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"One of the CSV files is empty: {e}") from e
        except Exception as e:
            raise Exception(f"Error occurred while loading static data: {e}") from e

if __name__ == "__main__":
    try:
        # Load static and dynamic data with specified parameters
        resolution = "50km"
        static_data = DataLoader.load_static_data(resolution=resolution)
        fire_data, weather_data = DataLoader.load_dynamic_data(start_date=pd.to_datetime('2020-01-01').date(), 
                                                               end_date=pd.to_datetime('2022-12-31').date(), 
                                                               resolution=resolution)

        # Print information about loaded datasets
        print("Fire Data - Min Date:", fire_data['ACQ_DATE'].min(), "Max Date:", fire_data['ACQ_DATE'].max())
        print("Weather Data - Min Date:", weather_data['ACQ_DATE'].min(), "Max Date:", weather_data['ACQ_DATE'].max())
        print("Static Data - Shape:", static_data.shape)
        print("Fire Data - Shape:", fire_data.shape)
        print("Weather Data - Shape:", weather_data.shape)
    
    except Exception as e:
        print(f"An error occurred: {e}")