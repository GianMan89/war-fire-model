def round_lat_lon(lat, lon):
    """
    Rounds latitude and longitude to the nearest .5 or .0.

    Parameters:
    lat (float): Latitude value.
    lon (float): Longitude value.

    Returns:
    tuple: Rounded latitude and longitude.
    """
    def round_to_half(value):
        return round(value * 2) / 2

    rounded_lat = round_to_half(lat)
    rounded_lon = round_to_half(lon)
    
    return rounded_lat, rounded_lon