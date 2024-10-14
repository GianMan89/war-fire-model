import dash
from dash import dcc, html, dash_table
import dash_leaflet as dl
import pandas as pd
import geopandas as gpd
from dash.dependencies import Input, Output, State
import json
import numpy as np
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

# Load data
fires_data = pd.read_csv('results/50km/test_predictions.csv')

# Convert ACQ_DATE to datetime.date format for date picker compatibility
fires_data['ACQ_DATE'] = pd.to_datetime(fires_data['ACQ_DATE']).dt.date

# Create a GeoDataFrame from the fires data
fires_gdf = gpd.GeoDataFrame(fires_data, geometry=gpd.points_from_xy(fires_data.LONGITUDE, fires_data.LATITUDE))

# Load Ukraine borders
ukraine_borders = gpd.read_file('data/ukr_borders/ukr_borders.shp')

# Load Russian-occupied territories
rus_control = gpd.read_file('data/rus_control/rus_control_2023.shp')
rus_control.to_crs(epsg=4326, inplace=True)
rus_control = rus_control.drop(columns=['CreationDa', 'EditDate'])

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Ukraine Forest Fires Dashboard"

# Map of Ukraine
ukraine_center = [48.3794, 31.1656]

# Get min and max dates for the slider
min_date = fires_gdf['ACQ_DATE'].min()
max_date = fires_gdf['ACQ_DATE'].max()

# Generate slider marks for every second month
slider_marks = {i: (min_date + pd.DateOffset(days=i)).strftime('%m-%Y') for i in range(0, (max_date - min_date).days + 1, 60)}

# Layout
app.layout = html.Div([
    dl.Map(id='fire-map', center=ukraine_center, zoom=6, children=[
        dl.TileLayer(url='https://{s}.tile.openstreetmap.de/{z}/{x}/{y}.png',
                     attribution='Map data © OpenStreetMap contributors', detectRetina=True),
        dl.LayerGroup(id='fire-layer', children=[]),
        dl.LayerGroup(id='ukraine-borders-layer', children=[
            dl.GeoJSON(data=json.loads(ukraine_borders.to_json()),
                       options=dict(style=dict(color='black', weight=3, opacity=1.0)))
        ]),
        dl.LayerGroup(id='rus-control-layer', children=[
            dl.GeoJSON(data=json.loads(rus_control.to_json()),
                       options=dict(style=dict(color='red', weight=2, fill=True, fillColor='red', fillOpacity=0.3, dashArray='5, 5')))
        ])
    ], style={"width": "100vw", "height": "100vh", "position": "absolute", "top": 0, "left": 0, "zIndex": 1}),

    html.Div([
        html.Label("Settings", style={"font-weight": "bold", "font-size": "16px"}),
        html.Button("Toggle Ukraine Borders", id='toggle-ukraine-borders', n_clicks=1, style={"margin-top": "10px"}),
        html.Button("Toggle Russian-Occupied Areas", id='toggle-rus-control', n_clicks=1, style={"margin-top": "10px"}),
    ], style={"position": "absolute", "top": "10px", "right": "10px", "background-color": "#ffffff", "padding": "20px", "border-radius": "10px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.1)", "zIndex": 2}),

    html.Div([
        html.Label("Select Date", style={"font-weight": "bold", "font-size": "16px"}),
        dcc.Slider(
            id='start-date-slider',
            min=0,
            max=(max_date - min_date).days,
            value=0,
            marks=slider_marks,
            tooltip={"placement": "bottom", "always_visible": True},
            step=1
        ),
        html.Div(id='selected-date', style={"margin-top": "10px", "font-weight": "bold", "font-size": "16px"})
    ], style={"position": "absolute", "bottom": "10px", "left": "5%", "right": "5%", "background-color": "#ffffff", "padding": "10px", "border-radius": "10px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.1)", "zIndex": 2}),

    html.Div([
        dash_table.DataTable(
            id='fire-details-table',
            columns=[
                {'name': 'Attribute', 'id': 'attribute'},
                {'name': 'Value', 'id': 'value'}
            ],
            style_table={'width': '100%', 'margin': '0 auto', 'border': '1px solid #ddd', 'box-shadow': '0px 4px 10px rgba(0, 0, 0, 0.1)'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f9f9f9'},
            data=[],
        )
    ], style={"position": "absolute", "top": "10px", "left": "10px", "background-color": "#ffffff", "padding": "10px", "border-radius": "10px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.1)", "zIndex": 2, "display": "none"}, id='fire-details-container')
])

# Fire markers colored by their label
def generate_fire_markers(data):
    markers = []
    for _, row in data.iterrows():
        markers.append(dl.CircleMarker(
            center=[row.geometry.y, row.geometry.x],
            radius=2,
            color='red',
            fill=True,
            fillOpacity=0.6,
            id={'type': 'fire-marker', 'index': row.name}
        ))
    return markers

# Load fires only once based on the selected date range
@app.callback(
    [Output('fire-layer', 'children'),
     Output('selected-date', 'children')],
    [Input('start-date-slider', 'value'),
     Input('fire-map', 'bounds')]
)
def update_fire_layer(start_date_offset, bounds):
    if not bounds or len(bounds) < 2:
        bounds = [[44.0, 22.0], [52.0, 40.0]]  # Approximate bounds for Ukraine
    
    start_date = min_date + pd.Timedelta(days=start_date_offset)
    end_date = start_date  # Show data for only one day

    # Extract bounds
    south_west = bounds[0]
    north_east = bounds[1]

    # Filter data based on the date range and the current map bounds
    filtered_data = fires_gdf[(fires_gdf['ACQ_DATE'] == start_date) &
                              (fires_gdf['LATITUDE'] >= south_west[0]) & (fires_gdf['LATITUDE'] <= north_east[0]) &
                              (fires_gdf['LONGITUDE'] >= south_west[1]) & (fires_gdf['LONGITUDE'] <= north_east[1])]
    selected_date_str = f"Selected Date: {start_date.strftime('%d-%m-%Y')}"
    return generate_fire_markers(filtered_data), selected_date_str

# Toggle Ukraine borders and Russian-occupied areas
@app.callback(
    [Output('ukraine-borders-layer', 'children'),
     Output('rus-control-layer', 'children')],
    [Input('toggle-ukraine-borders', 'n_clicks'),
     Input('toggle-rus-control', 'n_clicks')]
)
def toggle_layers(toggle_borders_clicks, toggle_rus_control_clicks):
    borders_layer = []
    rus_control_layer = []

    if toggle_borders_clicks % 2 == 1:
        borders_layer = [
            dl.GeoJSON(data=json.loads(ukraine_borders.to_json()),
                       options=dict(style=dict(color='black', weight=3, opacity=1.0)))
        ]
    if toggle_rus_control_clicks % 2 == 1:
        rus_control_layer = [
            dl.GeoJSON(data=json.loads(rus_control.to_json()),
                       options=dict(style=dict(color='red', weight=2, fill=True, fillColor='red', fillOpacity=0.3, dashArray='5, 5')))
        ]
    
    return borders_layer, rus_control_layer

# Update the table with fire details based on map click
@app.callback(
    [Output('fire-details-table', 'data'),
     Output('fire-details-container', 'style')],
    [Input('fire-map', 'click_lat_lng')],
    prevent_initial_call=True
)
def update_fire_details(click_lat_lng):
    if not click_lat_lng:
        return [], {'display': 'none'}
    lat, lon = click_lat_lng
    radius = 0.5  # Radius in degrees for selecting nearby fires
    selected_fires = fires_gdf[(fires_gdf['LATITUDE'] >= lat - radius) & (fires_gdf['LATITUDE'] <= lat + radius) &
                               (fires_gdf['LONGITUDE'] >= lon - radius) & (fires_gdf['LONGITUDE'] <= lon + radius)]
    
    data = []
    for _, row in selected_fires.iterrows():
        data.append({
            'attribute': 'Date', 'value': str(row['ACQ_DATE'])
        })
        data.append({
            'attribute': 'Latitude', 'value': row['LATITUDE']
        })
        data.append({
            'attribute': 'Longitude', 'value': row['LONGITUDE']
        })
        data.append({
            'attribute': 'Significance Score', 'value': row['SIGNIFICANCE_SCORE_DECAY']
        })
        data.append({
            'attribute': 'Abnormal Label', 'value': row['ABNORMAL_LABEL_DECAY']
        })
    
    return data, {"position": "absolute", "top": "10px", "left": "10px", "background-color": "#ffffff", "padding": "10px", "border-radius": "10px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.1)", "zIndex": 2, "display": "block"}

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)