# Define the paths
paths = {
    "data_dir": "data",
    "scripts_dir": "src",
    "models_dir": "models",
    "results_dir": "results",
    "fire_data_dir": "data/ukr_fires",
    "weather_data_dir": "data/ukr_weather",
    "land_use_data_dir": "data/ukr_land_use",
    "oblasts_data_dir": "data/ukr_oblasts",
    "pop_density_data_dir": "data/ukr_pop_density",
    "border_data_dir": "data/ukr_borders",
    }

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