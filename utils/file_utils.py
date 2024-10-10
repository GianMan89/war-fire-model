import joblib
import os

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
    Retrieve the path associated with the given path identifier.

    Parameters
    ----------
    path_id : str
        The identifier for the desired path.
    Returns
    -------
    str
        The path corresponding to the provided path identifier.
    Raises
    ------
    KeyError
        If the provided path_id does not exist in the paths dictionary.
    """
    if path_id not in paths:
        raise KeyError(f"Path identifier {path_id} does not exist.")
    return paths[path_id]


def save_large_model(model, base_filename, chunk_size=100):
    """
    Saves a large model by splitting it into smaller chunks.

    Parameters
    ----------
    model : object
        Trained model to be saved.
    base_filename : str
        Base name for the saved files.
    chunk_size : int, optional, default=100
        Maximum chunk size in MB.
    Returns
    -------
    int
        Number of parts the model was split into.
    """
    # Serialize the model to bytes
    model_bytes = joblib.dump(model, None)
    
    # Split bytes into smaller parts
    total_size = len(model_bytes[0])
    part_number = 0
    start = 0
    chunk_size_bytes = chunk_size * 1024 * 1024  # Convert MB to Bytes

    while start < total_size:
        end = start + chunk_size_bytes
        chunk = model_bytes[0][start:end]
        chunk_filename = f"{base_filename}_part_{part_number}.pkl"
        
        # Save the chunk
        with open(chunk_filename, 'wb') as file:
            file.write(chunk)
        
        start = end
        part_number += 1

    print(f"Model saved in {part_number} parts, each <= {chunk_size} MB.")
    return part_number

def load_large_model(base_filename, total_parts):
    """
    Load a large model that was split into smaller chunks.

    Parameters
    ----------
    base_filename : str
        Base name for the saved files.
    total_parts : int
        Number of parts the model was split into.
    Returns
    -------
    model : object
    """
    model_bytes = b""
    
    for part_number in range(total_parts):
        chunk_filename = f"{base_filename}_part_{part_number}.pkl"
        with open(chunk_filename, 'rb') as file:
            model_bytes += file.read()

    # Deserialize bytes into the model
    model = joblib.load(model_bytes)
    return model


# Example usage
if __name__ == "__main__":
    print(get_path("data_dir"))
    # save_large_model(trained_model, "my_model", chunk_size=100)
    # model = load_large_model("my_model", total_parts=14)