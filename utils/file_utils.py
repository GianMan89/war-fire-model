import os
import tempfile
import dill # joblib cannot serialize lambda functions
from split_file_writer import SplitFileWriter
from split_file_reader import SplitFileReader
from io import BytesIO
import glob

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

def save_large_model(model, base_filename, part_size=100):
    """
    Save a large model by splitting it into smaller parts.
    This function saves a large model by first dumping it into a temporary file
    and then splitting this file into smaller parts of specified size. The parts
    are saved with filenames based on the provided base filename.
    Parameters
    ----------
    model : object
        The model to be saved. This can be any object that is serializable by joblib.
    base_filename : str
        The base filename for the saved parts. Each part will have this base filename
        followed by a part number.
    part_size : int, optional, default=100
        The maximum size of each part in megabytes (MB). The model will be split into
        parts of this size.
    Returns
    -------
    int
        Number of parts the model was split into.
    """
    # Create a temporary file to store the model
    with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
        dill.dump(model, temp_model_file)
        temp_model_path = temp_model_file.name
    
    # Split the saved model file into parts
    part_size_bytes = part_size * 1024 * 1024  # Convert MB to bytes
    part_number = 0
    with open(temp_model_path, 'rb') as infile, SplitFileWriter(f"{base_filename}_part.", part_size_bytes) as sfw:
        for chunk in iter(lambda: infile.read(part_size_bytes), b''):
            sfw.write(chunk)
            part_number += 1
    
    # Remove the temporary file
    os.remove(temp_model_path)
    print(f"Model saved in {part_number} parts, each <= {part_size} MB.")
    return part_number

def load_large_model(base_filename, num_parts=None):
    """
    Load a large model that has been split into multiple parts.
    Parameters
    ----------
    base_filename : str
        The base filename of the split model parts. Each part file is expected
        to follow the naming convention `{base_filename}_part.XXX` where `XXX`
        is a zero-padded part number.
    num_parts : int
        The number of parts the model has been split into. If not provided, the
        function will attempt to determine this automatically by looking for
        part files matching the base filename pattern. Default is None.
    Returns
    -------
    model : object
        The loaded model object.
    Notes
    -----
    This function uses `SplitFileReader` to read the split parts as a binary-like
    file and then loads the model using `joblib.load`.
    Examples
    --------
    >>> model = load_large_model('model_checkpoint', 5)
    """
    if num_parts is None:
        # Check how many parts the model was split into
        # Get the list of files matching the base filename pattern
        part_files = glob.glob(f"{base_filename}_part.*")
        # Count the number of parts
        num_parts = len(part_files)
    
    # Define the list of split part files
    filepaths = [f"{base_filename}_part.{i:03d}" for i in range(num_parts)]
    
    # Use SplitFileReader to read the model parts as binary
    with SplitFileReader(filepaths) as sfr:
        # Load the model from the split parts, which are read as a binary-like file
        model_bytes = sfr.read()  # Read all parts into a single bytes object
        model = dill.load(BytesIO(model_bytes))

    return model


if __name__ == "__main__":
    # Example usage
    print(get_path("data_dir"))
    # num_parts = save_large_model(trained_model, "my_model", part_size=100)
    # loaded_model = load_large_model("my_model", num_parts=num_parts)