import pandas as pd

# Define the parameters
parameters = {
    "border_epsg": 4326,
    "nasa_firms_api_url": "https://firms.modaps.eosdis.nasa.gov/api/country/csv",
    "weather_api_url": "https://archive-api.open-meteo.com/v1/archive",
    "satellites": ["VIIRS_NOAA20_NRT", "MODIS_NRT"],
    "decay_rate": 1.0, 
    "midpoint": 5, 
    "cutoff": 10,
    "quantile": 0.95,
    "n_components": 0.95,
    "calibration_confidence": 0.95,
    "n_estimators": 100,
    "random_state": 42,
    }

# Load the MAP_KEY for the NASE FIRMS API
try:
    with open('config/nasa_firms_api_key.txt', 'r') as file:
        MAP_KEY = file.read().strip()
except FileNotFoundError:
    print("The file nasa_firms_api_key.txt was not found. Please provide a valid API key.")
    MAP_KEY = input("Enter your NASA FIRMS API key: ")

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
    print(get_parameter("n_components"))