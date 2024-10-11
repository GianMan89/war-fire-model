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
    """
    
    def round_to_1km(value):
        return round(value, 2)
    
    def round_to_10km(value):
        return round(value, 1)

    def round_to_50km(value):
        return round(value * 2) / 2

    if resolution == "1km":
        round_fn = round_to_1km
    elif resolution == "10km":
        round_fn = round_to_10km
    elif resolution == "50km":
        round_fn = round_to_50km
    else:
        raise ValueError("Invalid resolution. Possible values are '1km', '10km', and '50km'.")
    
    rounded_lat = round_fn(lat)
    rounded_lon = round_fn(lon)
    
    return rounded_lat, rounded_lon