import pandas as pd

# Define the paths
paths = {
    "data_dir": "data",
    "scripts_dir": "scripts",
    "models_dir": "models",
    "results_dir": "results",
    "fire_data_dir": "data/ukr_fires",
    "weather_data_dir": "data/ukr_weather",
    "nasa_firms_api_url": "https://firms.modaps.eosdis.nasa.gov/api/country/csv",
    "weather_api_url": "https://archive-api.open-meteo.com/v1/archive",
    "satellites": ["VIIRS_NOAA20_NRT", "MODIS_NRT"],
    }

# Load the MAP_KEY from the nasa_firms_api_key.txt file
with open('config/nasa_firms_api_key.txt', 'r') as file:
    MAP_KEY = file.read().strip()

# Add the MAP_KEY to the paths dictionary
paths["nasa_firms_api_key"] = MAP_KEY

# Load the ukr_weather_api_call_data.csv file into the paths dictionary
ukr_weather_api_call_data = pd.read_csv("config/ukr_weather_api_call_data.csv")
paths["ukr_weather_api_call_data"] = ukr_weather_api_call_data

def get_path(path_id):
    """
    Retrieve the path associated with the given ID from the paths dictionary.
    Args:
        path_id (str): The ID of the path to retrieve.
    Returns:
        str: The path corresponding to the provided ID.
    """
    return paths[path_id]

# Example usage
if __name__ == "__main__":
    print(get_path("data_dir"))