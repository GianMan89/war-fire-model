import pandas as pd
import os
import sys
import openmeteo_requests as omr
import requests_cache
from retry_requests import retry

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
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    sys.exit(1)

class DataUpdater:
    """
    The class fetches the latest data from the NASA FIRMS and Open-Meteo APIs,
    and saves it to CSV files.

    Attributes
    ----------
    api_url_1 : str
        The URL for the NASA FIRMS API.
    api_url_2 : str
        The URL for the weather API.
    api_key_1 : str
        The API key for the NASA FIRMS API.
    satellites : list
        A list of satellite names to fetch data from.
    fire_data_dir : str
        The directory where fire data CSV files are stored.
    weather_data_dir : str
        The directory where weather data CSV files are stored.
    current_date : datetime
        The current date for data fetching and updating.
    country_code : str
        The country code for the data to be fetched.
        Default is 'UKR' for Ukraine.
    
    Methods
    -------
    fetch_fire_data(newest_date)
    fetch_weather_data(newest_date)
    get_newest_date_from_csv(folder_path)
    save_data_to_csv(data, folder_path)
    update_data()
    """
    
    def __init__(self):
        try:
            # Initialize attributes using configuration parameters
            self.api_url_1 = get_parameter("nasa_firms_api_url")
            self.api_url_2 = get_parameter("weather_api_url")
            self.api_key_1 = get_parameter("nasa_firms_api_key")
            self.satellites = get_parameter("satellites")
            self.fire_data_dir = get_path("fire_data_dir")
            self.weather_data_dir = get_path("weather_data_dir")
            self.current_date = pd.to_datetime("today")
            self.country_code = "UKR"
        except KeyError as e:
            raise RuntimeError(f"Error initializing DataUpdater: {e}")
        except Exception as e:
            raise RuntimeError(f"Error initializing DataUpdater: {e}")

    def fetch_fire_data(self, newest_date):
        """
        Fetches fire data from the NASA FIRMS API for a specified country and date range.
        
        Parameters
        ----------
        newest_date : datetime
            The most recent date for which data is available.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the concatenated fire data from all satellites.

        Raises
        ------
        Exception
            If an error occurs during data fetching.
        """
        try:
            # Calculate the number of days between the newest date and the current date
            start_dates = [None]
            date_diff = (self.current_date - newest_date).days + pd.Timedelta(days=1).days
            if date_diff > 10:
                start_dates = [newest_date + pd.Timedelta(days=i) for i in range(0, date_diff, 10)]
                date_diff = 10
            print("Fetching data for the last", date_diff * len(start_dates), "days")
            
            # Iterate over the satellites and fetch the data
            dfs = []
            for satellite in self.satellites:
                if start_dates[0] is not None:
                    for start_date in start_dates:
                        # Construct API URL and fetch the data
                        url = f"{self.api_url_1}/{self.api_key_1}/{satellite}/{self.country_code}/{date_diff}/{start_date.strftime('%Y-%m-%d')}"
                        try:
                            dfs.append(pd.read_csv(url))
                        except Exception as e:
                            print(f"Error fetching fire data for satellite {satellite} and start date {start_date}: {e}")
                else:
                    url = f"{self.api_url_1}/{self.api_key_1}/{satellite}/{self.country_code}/{date_diff}"
                    try:
                        dfs.append(pd.read_csv(url))
                    except Exception as e:
                        print(f"Error fetching fire data for satellite {satellite}: {e}")
            
            # Select only LATITUDE, LONGITUDE, ACQ_DATE columns
            for i in range(len(dfs)):
                dfs[i].columns = map(str.upper, dfs[i].columns)
                dfs[i] = dfs[i][['LATITUDE', 'LONGITUDE', 'ACQ_DATE']]

            # Concatenate the dataframes
            return pd.concat(dfs)
        except Exception as e:
            print(f"Error in fetch_fire_data: {e}")
            return pd.DataFrame()

    def fetch_weather_data(self, newest_date):
        """
        Fetch weather data from the Open-Meteo API for specified locations and dates.

        Parameters
        ----------
        newest_date : datetime
            The most recent date for which data is available.
        
        Returns
        -------
        DataFrame
            A pandas DataFrame containing the merged hourly and daily weather data for each location.

        Raises
        ------
        Exception
            If an error occurs during data fetching.
        """
        try:
            # Setup the Open-Meteo API client with cache and retry on error
            cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = omr.Client(session=retry_session)
            
            # Get the ukr_weather_api_call_data from the configuration
            ukr_weather_api_call_data = get_parameter("ukr_weather_api_call_data")

            # Iterate over the rows in the ukr_weather_api_call_data
            dfs = []
            for index, row in ukr_weather_api_call_data.iterrows():
                latitude = row['LATITUDE']
                longitude = row['LONGITUDE']
                oblast_id = row['OBLAST_ID']

                # Define parameters for the API call
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": (newest_date - pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                    "end_date": self.current_date.strftime('%Y-%m-%d'),
                    "hourly": "cloud_cover",
                    "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", 
                              "rain_sum", "snowfall_sum", "wind_direction_10m_dominant"],
                    "timezone": "Europe/Moscow"
                }

                try:
                    responses = openmeteo.weather_api(self.api_url_2, params=params, verify=False)
                    response = responses[0]

                    # Process hourly data
                    hourly = response.Hourly()
                    hourly_cloud_cover = hourly.Variables(0).ValuesAsNumpy()
                    hourly_data = {"ACQ_DATE": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    )}
                    hourly_data["CLOUD_COVER (%)"] = hourly_cloud_cover
                    hourly_dataframe = pd.DataFrame(data=hourly_data)
                    
                    # Resample hourly data to daily means
                    hourly_dataframe.set_index('ACQ_DATE', inplace=True)
                    daily_means = hourly_dataframe.resample('D').mean().reset_index()
                    daily_means['ACQ_DATE'] = daily_means['ACQ_DATE'].dt.date
                    daily_means['OBLAST_ID'] = oblast_id

                    # Process daily data
                    daily = response.Daily()
                    daily_data = {
                        "ACQ_DATE": pd.date_range(
                            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                            freq=pd.Timedelta(seconds=daily.Interval()),
                            inclusive="left"
                        ),
                        "TEMPERATURE_2M_MAX (°C)": daily.Variables(0).ValuesAsNumpy(),
                        "TEMPERATURE_2M_MIN (°C)": daily.Variables(1).ValuesAsNumpy(),
                        "TEMPERATURE_2M_MEAN (°C)": daily.Variables(2).ValuesAsNumpy(),
                        "RAIN_SUM (MM)": daily.Variables(3).ValuesAsNumpy(),
                        "SNOWFALL_SUM (CM)": daily.Variables(4).ValuesAsNumpy(),
                        "WIND_DIRECTION_10M_DOMINANT (°)": daily.Variables(5).ValuesAsNumpy()
                    }
                    daily_dataframe = pd.DataFrame(data=daily_data)
                    daily_dataframe['ACQ_DATE'] = daily_dataframe['ACQ_DATE'].dt.date

                    # Merge hourly and daily data
                    merged_data = daily_dataframe.merge(daily_means, on='ACQ_DATE', how='left')
                    merged_data['OBLAST_ID'] = oblast_id

                    # Reorder columns
                    ordered_columns = [
                        "OBLAST_ID", "ACQ_DATE", "TEMPERATURE_2M_MAX (°C)", "TEMPERATURE_2M_MIN (°C)",
                        "TEMPERATURE_2M_MEAN (°C)", "RAIN_SUM (MM)", "SNOWFALL_SUM (CM)",
                        "WIND_DIRECTION_10M_DOMINANT (°)", "CLOUD_COVER (%)"
                    ]
                    dfs.append(merged_data[ordered_columns])

                except Exception as e:
                    print(f"Error fetching weather data for oblast {oblast_id}: {e}")

            return pd.concat(dfs)
        except Exception as e:
            print(f"Error in fetch_weather_data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_newest_date_from_csv(folder_path):
        """
        Retrieves the newest date from CSV files in the specified folder.
        
        Parameters
        ----------
        folder_path : str
            The path to the folder containing the CSV files.
        
        Returns
        -------
        datetime or None
            The most recent date found in the 'ACQ_DATE' column of the CSV files.

        Raises
        ------
        Exception
            If an error occurs during date retrieval.
        """
        try:
            newest_date = None
            
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_csv(file_path)
                    if 'ACQ_DATE' in df.columns:
                        df['ACQ_DATE'] = pd.to_datetime(df['ACQ_DATE'])
                        max_date = df['ACQ_DATE'].max()
                        if newest_date is None or max_date > newest_date:
                            newest_date = max_date
            
            return newest_date
        except Exception as e:
            print(f"Error in get_newest_date_from_csv: {e}")
            return None

    @staticmethod
    def save_data_to_csv(data, folder_path):
        """
        Saves the new data to CSV files, separated by year. If a file for a particular year already exists,
        concatenates the new data with the existing data. If not, creates a new file for that year.
        Ensures that the new data is sorted by day and has capitalized column names.
        
        Parameters
        ----------
        data : pd.DataFrame
            The new data to be saved.
        folder_path : str
            The path to the folder where the CSV files will be saved.
        
        Returns
        -------
        None

        Raises
        ------
        Exception
            If an error occurs during data saving.
        """
        try:
            # Ensure column names are capitalized
            data.columns = [col.upper() for col in data.columns]
            # Add a 'YEAR' column to the data
            data['ACQ_DATE'] = pd.to_datetime(data['ACQ_DATE'])
            data['YEAR'] = data['ACQ_DATE'].dt.year
            # Group the data by year
            grouped = data.groupby('YEAR')
            
            for year, group in grouped:
                try:
                    # Search for a file which ends with the year
                    existing_files = [f for f in os.listdir(folder_path) if f.endswith(f"{year}.csv")]
                    if existing_files:
                        # If the file exists, read the existing data and concatenate
                        file_path = os.path.join(folder_path, existing_files[0])
                        existing_data = pd.read_csv(file_path)
                        existing_data.drop_duplicates(inplace=True)
                        existing_data['ACQ_DATE'] = pd.to_datetime(existing_data['ACQ_DATE'])
                        group = group[group.columns.intersection(existing_data.columns)]
                        if 'OBLAST_ID' in group.columns:
                            group = pd.concat([existing_data, group]).drop_duplicates(subset=['ACQ_DATE', 'OBLAST_ID'], keep='last')
                        else:
                            group = pd.concat([existing_data, group])
                    else:
                        # If the file does not exist, create a new file path
                        existing_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
                        if existing_files:
                            base_file_name = existing_files[0].rsplit('_', 1)[0]
                            file_path = os.path.join(folder_path, f"{base_file_name}_{year}.csv")
                        else:
                            file_path = os.path.join(folder_path, f"data_{year}.csv")

                    # Drop any duplicates
                    group.drop_duplicates(inplace=True)
                    # Sort the data by 'ACQ_DATE'
                    group = group.sort_values(by='ACQ_DATE')
                    # Save the data to CSV
                    group.to_csv(file_path, index=False)
                    # Reload the data from the file and make sure no duplicates are present
                    df = pd.read_csv(file_path)
                    df.drop_duplicates(inplace=True)
                    df.to_csv(file_path, index=False)
                    print(f"Saved data to {file_path}")
                except Exception as e:
                    print(f"Error saving data for year {year}: {e}")
        except Exception as e:
            print(f"Error in save_data_to_csv: {e}")

    def update_data(self):
        """
        Updates the fire and weather data by fetching the latest data and saving it to CSV files.

        This method performs the following steps:
        1. Checks the newest date in the existing fire data CSV files.
        2. If the fire data is not up to date, fetches the latest fire data and saves it to CSV.
        3. Checks the newest date in the existing weather data CSV files.
        4. If the weather data is not up to date, fetches the latest weather data and saves it to CSV.

        Returns
        -------
        self : DataUpdater
            The instance of the class.

        Raises
        ------
        ValueError
            If no existing fire data is found.
        Exception
            If an error occurs during data updating.
        """
        try:
            # Update the fire data
            newest_date_fire = self.get_newest_date_from_csv(self.fire_data_dir)
            if newest_date_fire is None:
                raise ValueError("No existing fire data found.")
            print("Newest date for fire data:", newest_date_fire)
            if (self.current_date - newest_date_fire).days <= 0:
                print("Fire data is already up to date.")
            else:
                fire_data = self.fetch_fire_data(newest_date_fire)
                if fire_data.empty:
                    print("No new fire data was fetched.")
                else:
                    print("Fetched fire data with shape:", fire_data.shape)
                    self.save_data_to_csv(fire_data, self.fire_data_dir)
                    print("Updated fire data date is:", self.get_newest_date_from_csv(self.fire_data_dir))

            # Update the weather data
            newest_date_weather = self.get_newest_date_from_csv(self.weather_data_dir)
            if newest_date_weather is None:
                raise ValueError("No existing weather data found.")
            print("Newest date for weather data:", newest_date_weather)
            if (self.current_date - newest_date_weather).days <= 0:
                print("Weather data is already up to date.")
            else:
                weather_data = self.fetch_weather_data(newest_date_weather)
                if weather_data.empty:
                    print("No new weather data was fetched.")
                else:
                    print("Fetched weather data:", weather_data.shape)
                    self.save_data_to_csv(weather_data, self.weather_data_dir)
                    print("Updated weather data date is:", self.get_newest_date_from_csv(self.weather_data_dir))

        except Exception as e:
            print(f"Error in update_data: {e}")

        return self

if __name__ == "__main__":
    updater = DataUpdater()
    updater.update_data()