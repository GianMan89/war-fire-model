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


# Example usage
if __name__ == "__main__":
    try:
        lat, lon = 12.34567, 98.76543
        rounded_coords = round_lat_lon(lat, lon, resolution="10km")
        print("Rounded coordinates:", rounded_coords)
    except (TypeError, ValueError) as e:
        print("Error:", e)