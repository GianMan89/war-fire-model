import pandas as pd
import os
import sys
import openmeteo_requests as omr
import requests_cache
from retry_requests import retry

import warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

from utils.file_utils import get_path
from config.config import get_parameter


class DataUpdater:
    """
    A class to handle the updating of fire and weather data from various APIs.
    Attributes:
        api_url_1 (str): The URL for the NASA FIRMS API.
        api_url_2 (str): The URL for the weather API.
        api_key_1 (str): The API key for the NASA FIRMS API.
        satellites (list): A list of satellite names to fetch data from.
        fire_data_dir (str): The directory where fire data CSV files are stored.
        weather_data_dir (str): The directory where weather data CSV files are stored.
        current_date (datetime): The current date for data fetching and updating.
    Methods:
        fetch_fire_data(newest_date, current_date, country_code="UKR"):
            Fetches fire data from the NASA FIRMS API for a specified country and date range.
        fetch_weather_data():
            Fetches weather data from the Open-Meteo API for specified locations and dates.
        get_newest_date_from_csv(folder_path):
            Retrieves the newest date from CSV files in the specified folder.
        save_data_to_csv(data, folder_path):
            Saves the new data to CSV files, separated by year.
        update_data():
            Updates the fire and weather data by fetching the latest data and saving it to CSV files.
    """
    
    def __init__(self):
        self.api_url_1 = get_parameter("nasa_firms_api_url")
        self.api_url_2 = get_parameter("weather_api_url")
        self.api_key_1 = get_parameter("nasa_firms_api_key")
        self.satellites = get_parameter("satellites")
        self.fire_data_dir = get_path("fire_data_dir")
        self.weather_data_dir = get_path("weather_data_dir")
        self.current_date = pd.to_datetime("today")
        self.country_code = "UKR"

    def fetch_fire_data(self, newest_date):
        """
        Fetches fire data from the NASA FIRMS API for a specified country and date range.
        This method calculates the number of days between the newest and current dates,
        constructs the API URL for each satellite, and fetches the fire data for the 
        specified country and date range. The data from all satellites is then concatenated 
        into a single DataFrame.
        Args:
            newest_date (datetime): The most recent date for which data is available.
            country_code (str, optional): The country code for which the data is to be fetched. 
                                          Defaults to "UKR".
        Returns:
            pd.DataFrame: A DataFrame containing the concatenated fire data from all satellites.
        """
        # Get the distance between the newest date and the current date
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
                    # Load the last date_diff days of data
                    url = f"{self.api_url_1}/{self.api_key_1}/{satellite}/{self.country_code}/{date_diff}/{start_date.strftime('%Y-%m-%d')}"
                    # Fetch the data
                    dfs.append(pd.read_csv(url))
            else:
                # Load the last date_diff days of data
                url = f"{self.api_url_1}/{self.api_key_1}/{satellite}/{self.country_code}/{date_diff}"
                # Fetch the data
                dfs.append(pd.read_csv(url))
        
        # Select only LATITUDE,LONGITUDE,ACQ_DATE columns
        for i in range(len(dfs)):
            dfs[i].columns = map(str.upper, dfs[i].columns)
            dfs[i] = dfs[i][['LATITUDE', 'LONGITUDE', 'ACQ_DATE']]

        # Concatenate the dataframes
        return pd.concat(dfs)

    def fetch_weather_data(self, newest_date):
        """
        Fetches weather data from the Open-Meteo API for specified locations and dates.
        Args:
            newest_date (datetime): The most recent date for which data is available.
        Returns:
            DataFrame: A pandas DataFrame containing the merged hourly and daily weather data for each location.
        """
        
        
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = omr.Client(session = retry_session)
        # Get the ukr_weather_api_call_data from the paths dictionary
        ukr_weather_api_call_data = get_parameter("ukr_weather_api_call_data")

        # Iterate over the rows in the ukr_weather_api_call_data
        dfs = []
        for index, row in ukr_weather_api_call_data.iterrows():
            # Get the latitude and longitude from the row
            latitude = row['LATITUDE']
            longitude = row['LONGITUDE']
            oblast_id = row['OBLAST_ID']

            # Define the parameters for the API call
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": (newest_date - pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                "end_date": (self.current_date).strftime('%Y-%m-%d'),
                "hourly": "cloud_cover",
                "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", 
                          "rain_sum", "snowfall_sum", "wind_direction_10m_dominant"],
                "timezone": "Europe/Moscow"
            }
            responses = openmeteo.weather_api(self.api_url_2, params=params, verify=False)
            # Process first location. Add a for-loop for multiple locations or weather models
            response = responses[0]

            # Process hourly data. The order of variables needs to be the same as requested.
            hourly = response.Hourly()
            hourly_cloud_cover = hourly.Variables(0).ValuesAsNumpy()
            hourly_data = {"ACQ_DATE": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            )}
            hourly_data["CLOUD_COVER (%)"] = hourly_cloud_cover
            hourly_dataframe = pd.DataFrame(data = hourly_data)
            # Resample hourly data to daily means
            hourly_dataframe.set_index('ACQ_DATE', inplace=True)
            daily_means = hourly_dataframe.resample('D').mean().reset_index()
            daily_means['ACQ_DATE'] = daily_means['ACQ_DATE'].dt.date
            daily_means['OBLAST_ID'] = oblast_id

            # Process daily data. The order of variables needs to be the same as requested.
            daily = response.Daily()
            daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
            daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
            daily_temperature_2m_mean = daily.Variables(2).ValuesAsNumpy()
            daily_rain_sum = daily.Variables(3).ValuesAsNumpy()
            daily_snowfall_sum = daily.Variables(4).ValuesAsNumpy()
            daily_wind_direction_10m_dominant = daily.Variables(5).ValuesAsNumpy()
            daily_data = {"ACQ_DATE": pd.date_range(
                start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
                end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = daily.Interval()),
                inclusive = "left"
            )}
            daily_data["ACQ_DATE"] = daily_data["ACQ_DATE"].date
            daily_data["TEMPERATURE_2M_MAX (°C)"] = daily_temperature_2m_max
            daily_data["TEMPERATURE_2M_MIN (°C)"] = daily_temperature_2m_min
            daily_data["TEMPERATURE_2M_MEAN (°C)"] = daily_temperature_2m_mean
            daily_data["RAIN_SUM (MM)"] = daily_rain_sum
            daily_data["SNOWFALL_SUM (CM)"] = daily_snowfall_sum
            daily_data["WIND_DIRECTION_10M_DOMINANT (°)"] = daily_wind_direction_10m_dominant
            daily_dataframe = pd.DataFrame(data = daily_data)

            # Merge the hourly and daily data
            dfs.append(daily_dataframe.merge(daily_means, on = 'ACQ_DATE', how = 'left'))

            # Reorder the columns
            ordered_columns = [
                "OBLAST_ID", "ACQ_DATE", "TEMPERATURE_2M_MAX (°C)", "TEMPERATURE_2M_MIN (°C)",
                "TEMPERATURE_2M_MEAN (°C)", "RAIN_SUM (MM)", "SNOWFALL_SUM (CM)",
                "WIND_DIRECTION_10M_DOMINANT (°)", "CLOUD_COVER (%)"
            ]
            dfs[-1] = dfs[-1][ordered_columns]

        return pd.concat(dfs)
    
    @staticmethod
    def get_newest_date_from_csv(folder_path):
        """
        Retrieves the newest date from CSV files in the specified folder.
        This method iterates through all CSV files in the given folder, reads the 'ACQ_DATE' column,
        and returns the most recent date found across all files.
        Args:
            folder_path (str): The path to the folder containing the CSV files.
        Returns:
            datetime: The most recent date found in the 'ACQ_DATE' column of the CSV files.
                      Returns None if no valid dates are found.
        """
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

    @staticmethod
    def save_data_to_csv(data, folder_path):
        """
        Saves the new data to CSV files, separated by year. If a file for a particular year already exists,
        concatenates the new data with the existing data. If not, creates a new file for that year.
        Ensures that the new data is sorted by day and has capitalized column names.
        
        Args:
            data (pd.DataFrame): The new data to be saved.
            folder_path (str): The path to the folder where the CSV files will be saved.
        
        Returns:
            None
        """
        # Ensure column names are capitalized
        data.columns = [col.upper() for col in data.columns]
        # Add a 'YEAR' column to the data
        data['ACQ_DATE'] = pd.to_datetime(data['ACQ_DATE'])
        data['YEAR'] = data['ACQ_DATE'].dt.year
        # Group the data by year
        grouped = data.groupby('YEAR')
        
        for year, group in grouped:
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
                    group = pd.concat([existing_data, group]).drop_duplicates(subset=['ACQ_DATE', 
                                                                                      'OBLAST_ID'], keep='last')
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
        
        return None

    def update_data(self):
        """
        Updates the fire and weather data by fetching the latest data and saving it to CSV files.
        This method performs the following steps:
        1. Checks the newest date in the existing fire data CSV files.
        2. If the fire data is not up to date, fetches the latest fire data and saves it to CSV.
        3. Checks the newest date in the existing weather data CSV files.
        4. If the weather data is not up to date, fetches the latest weather data and saves it to CSV.

        Returns:
            self: The instance of the class.
        """
        
        # Update the fire data
        newest_date_fire = self.get_newest_date_from_csv(self.fire_data_dir)
        print("Newest date for fire data:", newest_date_fire)
        if (self.current_date - newest_date_fire).days <= 0:
            print("Fire data is already up to date.")
        else:
            fire_data = self.fetch_fire_data(newest_date_fire)
            print("Fetched fire data with shape:", fire_data.shape)
            self.save_data_to_csv(fire_data, self.fire_data_dir)
            print("Update fire data date is:", self.get_newest_date_from_csv(self.fire_data_dir))

        # Update the weather data
        newest_date_weather = self.get_newest_date_from_csv(self.weather_data_dir)
        print("Newest date for weather data:", newest_date_weather)
        if (self.current_date - newest_date_weather).days <= 0:
            print("Weather data is already up to date.")
        else:
            weather_data = self.fetch_weather_data(newest_date_weather)
            print("Fetched weather data:", weather_data.shape)
            self.save_data_to_csv(weather_data, self.weather_data_dir)
            print("Update weather data date is:", self.get_newest_date_from_csv(self.weather_data_dir))

        return self

def main():
    """
    Main function to demonstrate the usage of DataUpdater.
    """
    updater = DataUpdater()
    updater.update_data()

if __name__ == "__main__":
    main()