import os
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
    from utils.data_utils import force_datetime
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    sys.exit(1)

class FeatureEngineering:
    """
    FeatureEngineering class for processing and transforming fire data.
    
    Methods
    -------
    generate_fire_time_series(fire_data, start_date, end_date)
        Generates a time series dataset for fire data within grid cells.
    transform(fire_data, static_data, weather_data, start_date, end_date)
        Transforms the fire data by performing several operations.
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
            The fire data.
        start_date : datetime.date
            The start date of the data.
        end_date : datetime.date
            The end date of the data.

        Returns
        -------
        pd.DataFrame
            A concatenated DataFrame containing the time series data for all grid cells.

        Raises
        ------
        KeyError
            If a required column is missing in the fire data.
        RuntimeError
            If an error occurs during the generation of the fire time series.
        """
        try:
            # Enforce start and end date to be of type datetime.date
            start_date, end_date = force_datetime(start_date), force_datetime(end_date)
            time_series_data = {}
            for cell in fire_data['GRID_CELL'].unique():
                # Extract data for the current grid cell and drop irrelevant columns
                cell_data = fire_data[fire_data['GRID_CELL'] == cell].drop(columns=['FIRE_ID'])

                # Extract static data for each cell (values assumed constant across time)
                static_data = cell_data.iloc[0].drop(['ACQ_DATE', 'DAY_OF_YEAR', 'FIRE_COUNT_CELL'])

                # Set index to 'ACQ_DATE' for time-series operations
                cell_data.set_index('ACQ_DATE', inplace=True)
                cell_data.index = pd.to_datetime(cell_data.index)

                # Ensure no duplicate dates for the time series
                cell_data = cell_data[~cell_data.index.duplicated(keep='first')]

                # Reindex to fill missing dates with zero values
                cell_data = cell_data.reindex(pd.date_range(start=start_date, end=end_date, freq='D'), fill_value=0)

                # Add 'DAY_OF_YEAR' and reset index
                cell_data['DAY_OF_YEAR'] = cell_data.index.dayofyear
                cell_data['ACQ_DATE'] = cell_data.index
                cell_data.reset_index(drop=True, inplace=True)

                # Reorder columns to put 'ACQ_DATE' first
                cell_data = cell_data[['ACQ_DATE'] + [col for col in cell_data.columns if col != 'ACQ_DATE']]

                # Add static features back to the time series data
                for col in static_data.index:
                    cell_data[col] = static_data[col]

                # Store the processed time series data for the current grid cell
                time_series_data[cell] = cell_data

            # Concatenate all grid cell data into a single DataFrame
            time_series_data = pd.concat(time_series_data.values())
            return time_series_data

        except KeyError as e:
            raise KeyError(f"Missing expected column in fire_data: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while generating fire time series: {e}")
    
    def transform(self, fire_data, static_data, weather_data, start_date, end_date):
        """
        Transforms the fire data by performing several operations.

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
        X : pd.DataFrame
            The features of the data.
        y : pd.Series
            The target variable of the data.
        ids : pd.DataFrame
            The fire IDs and ACQ_DATE of the data.

        Raises
        ------
        KeyError
            If a required column is missing in the data.
        RuntimeError
            If an error occurs during the transformation.
        Exception
            If an unexpected error occurs.
        """
        try:
            # Sort the fire data by acquisition date
            fire_data.sort_values('ACQ_DATE', inplace=True)

            # Count the number of fires per grid cell per acquisition date
            fire_data['FIRE_COUNT_CELL'] = fire_data.groupby(['GRID_CELL', 'ACQ_DATE'])['ACQ_DATE'].transform('count')

            # Generate the fire time series
            time_series_data = self.generate_fire_time_series(fire_data, start_date, end_date)

            # Merge the time series data with static data
            time_series_data = pd.merge(time_series_data, static_data, how='left', on=['LATITUDE', 'LONGITUDE'])

            # Convert 'ACQ_DATE' to datetime.date format
            time_series_data['ACQ_DATE'] = time_series_data['ACQ_DATE'].apply(lambda x: pd.to_datetime(x).date())

            # Merge the time series data with weather data
            time_series_data = pd.merge(time_series_data, weather_data, how='left', on=['OBLAST_ID', 'ACQ_DATE'])

            # Split the data into features (X), target variable (y), and IDs
            X = time_series_data.drop(columns=['FIRE_COUNT_CELL', 'OBLAST_ID', 'ACQ_DATE', 'GRID_CELL', 
                                               'LATITUDE_ORIGINAL', 'LONGITUDE_ORIGINAL']).reset_index(drop=True)
            y = time_series_data['FIRE_COUNT_CELL'].reset_index(drop=True)
            ids = time_series_data[['ACQ_DATE', 'GRID_CELL']].reset_index(drop=True)

            return X, y, ids

        except KeyError as e:
            raise KeyError(f"Missing expected column in data: {e}")
        except pd.errors.MergeError as e:
            raise RuntimeError(f"Error while merging data: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during transformation: {e}")
