import pandas as pd

def round_lat_lon(lat, lon, resolution="50km"):
    """
    Rounds latitude and longitude to 1km, 10km, or 50km resolution.

    Parameters
    ----------
    lat : float
        Latitude value.
    lon : float
        Longitude value.
    resolution : str
        Resolution for rounding. Possible values are "1km", "10km", and "50km".

    Returns
    -------
    tuple
        Rounded latitude and longitude as a tuple (rounded_lat, rounded_lon).

    Raises
    ------
    TypeError
        If lat or lon is not a float or int.
    ValueError
        If resolution is not one of the accepted values.

    Examples
    --------
    >>> try:
    >>>    lat, lon = 12.34567, 98.76543
    >>>    rounded_coords = round_lat_lon(lat, lon, resolution="10km")
    >>>    print("Rounded coordinates:", rounded_coords)
    >>> except (TypeError, ValueError) as e:
    >>>    print("Error:", e)
    """
    
    # Check if latitude and longitude are of valid types (float or int)
    if not isinstance(lat, (float, int)):
        raise TypeError("Latitude must be a float or int.")
    if not isinstance(lon, (float, int)):
        raise TypeError("Longitude must be a float or int.")
    
    # Define rounding functions for different resolutions
    def round_to_1km(value):
        # Rounds value to 2 decimal places (approximately 1km resolution)
        return round(value, 2)
    
    def round_to_10km(value):
        # Rounds value to 1 decimal place (approximately 10km resolution)
        return round(value, 1)

    def round_to_50km(value):
        # Rounds value to the nearest 0.5 (approximately 50km resolution)
        return round(value * 2) / 2

    # Determine the rounding function based on the resolution parameter
    if resolution == "1km":
        round_fn = round_to_1km
    elif resolution == "10km":
        round_fn = round_to_10km
    elif resolution == "50km":
        round_fn = round_to_50km
    else:
        # Raise an error if an invalid resolution is provided
        raise ValueError("Invalid resolution. Possible values are '1km', '10km', and '50km'.")
    
    # Round latitude and longitude using the selected rounding function
    rounded_lat = round_fn(lat)
    rounded_lon = round_fn(lon)
    
    return rounded_lat, rounded_lon

def force_datetime(date):
    """
    Convert the provided date to a datetime.date object.

    Parameters
    ----------
    date : str, int, float, datetime-like, or list-like
        The date to be converted. This can be a string, integer, float, 
        datetime-like object, or a list-like object containing dates.
    
    Returns
    -------
    datetime.date
        The converted date as a datetime.date object.
    
    Raises
    ------
    ValueError
        If the provided date cannot be converted to a datetime.date object.
    """

    # Convert the provided date to datetime.date
    try:
        return pd.to_datetime(date).date()
    except Exception as e:
        raise ValueError(f"Invalid date format: {str(e)}")


# Example usage
if __name__ == "__main__":
    try:
        lat, lon = 12.34567, 98.76543
        rounded_coords = round_lat_lon(lat, lon, resolution="10km")
        print("Rounded coordinates:", rounded_coords)
    except (TypeError, ValueError) as e:
        print("Error:", e)
    
    try:
        date = '2023-01-01'
        date = force_datetime(date)
        print("Converted date:", date)
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")