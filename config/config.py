import pandas as pd

# Define the parameters
parameters = {
    "border_epsg": 4326,
    "nasa_firms_api_url": "https://firms.modaps.eosdis.nasa.gov/api/country/csv",
    "weather_api_url": "https://archive-api.open-meteo.com/v1/archive",
    "satellites": ["VIIRS_NOAA20_NRT", "MODIS_NRT"],
    }

# Load the MAP_KEY from the nasa_firms_api_key.txt file
with open('config/nasa_firms_api_key.txt', 'r') as file:
    MAP_KEY = file.read().strip()

# Add the MAP_KEY to the paths dictionary
parameters["nasa_firms_api_key"] = MAP_KEY

# Load the ukr_weather_api_call_data.csv file into the paths dictionary
ukr_weather_api_call_data = pd.read_csv("config/ukr_weather_api_call_data.csv")
parameters["ukr_weather_api_call_data"] = ukr_weather_api_call_data

def get_parameter(parameter_id):
    """
    Retrieve the parameter associated with the given ID from the parameter dictionary.
    Args:
        parameter_id (str): The ID of the parameter to retrieve.
    Returns:
        str: The parameter corresponding to the provided ID.
    """
    return parameters[parameter_id]

# Example usage
if __name__ == "__main__":
    print(get_parameter("data_dir"))