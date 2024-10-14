import dash
from dash import dcc, html
import dash_leaflet as dl
import pandas as pd
import geopandas as gpd
from dash.dependencies import Input, Output, State
import json
import numpy as np
import os
import sys
import warnings
from sklearn.cluster import KMeans

# Suppress warnings
warnings.filterwarnings("ignore")

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

# Load data
fires_data = pd.read_csv('results/50km/test_predictions.csv')



# Convert ACQ_DATE to datetime format for date picker compatibility
fires_data['ACQ_DATE'] = pd.to_datetime(fires_data['ACQ_DATE']).dt.date

# Create a GeoDataFrame from the fires data
fires_gdf = gpd.GeoDataFrame(fires_data, geometry=gpd.points_from_xy(fires_data.LONGITUDE, fires_data.LATITUDE))



# Sample the data to reduce the number of markers displayed (e.g., 0.1% of the data)
fires_gdf = fires_gdf.sample(frac=0.5, random_state=42)

# Load Ukraine borders
ukraine_borders = gpd.read_file('data/ukr_borders/ukr_borders.shp')

# Load Russian-occupied territories
rus_control = gpd.read_file('data/rus_control/rus_control_2023.shp')
rus_control.to_crs(epsg=4326, inplace=True)
rus_control = rus_control.drop(columns=['CreationDa', 'EditDate'])

# Initialize Dash app
app = dash.Dash(__name__)

# Map of Ukraine
ukraine_center = [48.3794, 31.1656]

# Get min and max dates for the slider
min_date = fires_gdf['ACQ_DATE'].min()
max_date = fires_gdf['ACQ_DATE'].max()

# Convert dates to ordinal for slider representation
min_date_ordinal = min_date.toordinal()
max_date_ordinal = max_date.toordinal()

# Layout
app.layout = html.Div([
    html.H1("Ukraine Forest Fires Dashboard"),
    html.P("Map of forest fires in Ukraine, colored by classification label."),
    dcc.Dropdown(
        id='date-dropdown',
        options=[
            {'label': pd.Timestamp(year=year, month=month, day=1).strftime('%B %Y'), 'value': pd.Timestamp(year=year, month=month, day=1).strftime('%Y-%m-%d')}
            for year in range(min_date.year, max_date.year + 1)
            for month in range(1, 13)
            if pd.Timestamp(year=year, month=month, day=1) >= pd.Timestamp(min_date) and pd.Timestamp(year=year, month=month, day=1) <= pd.Timestamp(max_date)
        ],
        value='2023-01-01',
        clearable=False
    ),
    html.Div(id='date-slider-labels', style={"text-align": "center", "margin-top": "10px"}),
    dl.Map(id='fire-map', center=ukraine_center, zoom=6, children=[
        dl.TileLayer(url='https://{s}.tile.openstreetmap.de/{z}/{x}/{y}.png',
                     attribution='Map data © OpenStreetMap contributors', detectRetina=True),
        # Add Ukraine border layer with a bold line
        dl.GeoJSON(data=json.loads(ukraine_borders.to_json()),
                   options=dict(style=dict(color='black', weight=3, opacity=1.0))),
        # Add Russian-occupied territories with red hatched design
        dl.GeoJSON(data=json.loads(rus_control.to_json()),
                   options=dict(style=dict(color='red', weight=2, fill=True, fillColor='red', fillOpacity=0.3, dashArray='5, 5'))),
        dl.LayerGroup(id='fire-layer', children=[])
    ], style={"width": "100%", "height": "600px"}),
    html.Div(id='fire-stats', style={"text-align": "center", "margin-top": "20px"})
])

# Fire markers colored by their label
def generate_fire_markers(data):
    markers = []
    # If there are more than 200 fires, cluster them to reduce the number of markers
    if len(data) > 200:
        num_clusters = 200  # Limit the number of clusters to 200
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        data['cluster'] = kmeans.fit_predict(data[['LATITUDE', 'LONGITUDE']])
    else:
        data['cluster'] = -1
    
    for cluster_id, group in data.groupby('cluster'):
        if cluster_id == -1 or len(group) == 1:  # Not clustered or single point
            for _, row in group.iterrows():
                color = 'red' if row['ABNORMAL_LABEL_DECAY'] else 'blue'
                significance_score = round(row['SIGNIFICANCE_SCORE_DECAY'], 3)
                tooltip_text = f"Date: {row['ACQ_DATE']}, Lat: {row['LATITUDE']}, Lon: {row['LONGITUDE']}, Score: {significance_score}"
                markers.append(dl.Marker(position=[row.geometry.y, row.geometry.x], children=dl.Tooltip(tooltip_text),
                                         icon=dict(iconUrl=f'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-{color}.png',
                                                   iconSize=[10, 15], iconAnchor=[5, 15], popupAnchor=[1, -14], tooltipAnchor=[6, -12])))
        else:  # Clustered points
            lat = group['LATITUDE'].mean()
            lon = group['LONGITUDE'].mean()
            count = len(group)
            color = 'red' if group['ABNORMAL_LABEL_DECAY'].iloc[0] else 'blue'
            tooltip_text = f"Cluster of {count} fires"
            icon_size = min(20, 10 + count * 2)  # Dynamically adjust icon size based on cluster size
            markers.append(dl.Marker(position=[lat, lon], children=dl.Tooltip(tooltip_text),
                                     icon=dict(iconUrl=f'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-{color}.png',
                                               iconSize=[icon_size, icon_size + 10], iconAnchor=[icon_size // 2, icon_size + 10], popupAnchor=[1, -icon_size - 9], tooltipAnchor=[8, -14])))
    return markers

# Update the map and statistics based on the selected date range
@app.callback(
    [Output('fire-layer', 'children'),
     Output('date-slider-labels', 'children'),
     Output('fire-stats', 'children')],
    [Input('date-dropdown', 'value'),
     Input('fire-map', 'zoom'),
     Input('fire-map', 'bounds')]
)
def update_fire_layer(date_range, zoom_level, bounds):
    data_to_use = fires_gdf
    start_date = pd.to_datetime(date_range).date()
    end_date = (pd.to_datetime(date_range) + pd.DateOffset(months=1) - pd.DateOffset(days=1)).date()
    
    # Extract bounds
    south_west = bounds[0]
    north_east = bounds[1]
    
    # Filter data based on the date range and the current map bounds
    filtered_data = fires_gdf[(fires_gdf['ACQ_DATE'] >= start_date) & (fires_gdf['ACQ_DATE'] <= end_date) &
                              (fires_gdf['LATITUDE'] >= south_west[0]) & (fires_gdf['LATITUDE'] <= north_east[0]) &
                              (fires_gdf['LONGITUDE'] >= south_west[1]) & (fires_gdf['LONGITUDE'] <= north_east[1]) &
                              (fires_gdf['ABNORMAL_LABEL_DECAY'] == 1)]
    
    labels = f"Start Date: {start_date} | End Date: {end_date}"
    
    # Calculate statistics
    total_fires = len(filtered_data)
    abnormal_fires = len(filtered_data.loc[filtered_data['ABNORMAL_LABEL_DECAY'] == 1])
    normal_fires = total_fires - abnormal_fires
    abnormal_percentage = (abnormal_fires / total_fires) * 100 if total_fires > 0 else 0
    normal_percentage = (normal_fires / total_fires) * 100 if total_fires > 0 else 0
    stats = f"Total Fires: {total_fires} | Abnormal Fires: {abnormal_fires} ({abnormal_percentage:.2f}%) | Normal Fires: {normal_fires} ({normal_percentage:.2f}%)"
    
    return generate_fire_markers(filtered_data), labels, stats

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)
    