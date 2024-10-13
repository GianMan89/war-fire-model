import pandas as pd

# Define the parameters dictionary to hold all required settings
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
    "n_neighbors": 10,
}

# Load the MAP_KEY for the NASA FIRMS API, with error handling
try:
    # Attempt to open and read the API key from the specified file
    with open('config/nasa_firms_api_key.txt', 'r') as file:
        MAP_KEY = file.read().strip()
except FileNotFoundError:
    # Handle the error if the file is not found
    print("The file nasa_firms_api_key.txt was not found. Please provide a valid API key.")
    MAP_KEY = input("Enter your NASA FIRMS API key: ")
except Exception as e:
    # Handle any other unforeseen errors during file reading
    print(f"An unexpected error occurred: {e}")
    MAP_KEY = input("Enter your NASA FIRMS API key: ")

# Add the MAP_KEY to the parameters dictionary
parameters["nasa_firms_api_key"] = MAP_KEY

# Load the ukr_weather_api_call_data.csv file into the parameters dictionary
try:
    # Attempt to load the CSV file containing weather API call data
    ukr_weather_api_call_data = pd.read_csv("config/ukr_weather_api_call_data.csv")
    parameters["ukr_weather_api_call_data"] = ukr_weather_api_call_data
except FileNotFoundError:
    # Handle the error if the file is not found
    print("The file ukr_weather_api_call_data.csv was not found. Please ensure the file exists in the 'config' directory.")
    parameters["ukr_weather_api_call_data"] = None
except pd.errors.EmptyDataError:
    # Handle the error if the file is empty
    print("The file ukr_weather_api_call_data.csv is empty. Please provide a valid CSV file.")
    parameters["ukr_weather_api_call_data"] = None
except pd.errors.ParserError:
    # Handle the error if the file cannot be parsed as a CSV
    print("The file ukr_weather_api_call_data.csv could not be parsed. Please check the file format.")
    parameters["ukr_weather_api_call_data"] = None
except Exception as e:
    # Handle any other unforeseen errors during file reading
    print(f"An unexpected error occurred while loading ukr_weather_api_call_data.csv: {e}")
    parameters["ukr_weather_api_call_data"] = None

def get_parameter(parameter_id):
    """
    Retrieve the parameter associated with the given ID from the parameters dictionary.
    
    Parameters
    ----------
    parameter_id : str
        The ID of the parameter to retrieve.

    Returns
    -------
    parameter
        The parameter corresponding to the provided ID, or None if the ID is not found.
    """
    try:
        # Attempt to retrieve the parameter from the dictionary
        return parameters[parameter_id]
    except KeyError:
        # Handle the case where the parameter ID does not exist
        print(f"The parameter ID '{parameter_id}' was not found in the parameters dictionary.")
        return None

# Example usage
if __name__ == "__main__":
    # Example usage to print the value of 'n_components' parameter
    try:
        n_components_value = get_parameter("n_components")
        if n_components_value is not None:
            print(n_components_value)
    except Exception as e:
        # Handle any unforeseen errors during retrieval or printing
        print(f"An error occurred during parameter retrieval: {e}")