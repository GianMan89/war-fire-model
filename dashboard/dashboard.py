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

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("Ukraine Forest Fires Dashboard", style={"text-align": "center", "margin-bottom": "20px"}),
        html.P("An interactive dashboard visualizing forest fires in Ukraine, categorized by classification labels. The dashboard allows users to explore spatial and temporal patterns of forest fires, helping to identify abnormal fire activity.",
               style={"text-align": "center", "margin-bottom": "30px", "font-size": "18px"}),
    ], style={"background-color": "#f9f9f9", "padding": "30px", "border-radius": "10px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.1)", "margin": "20px"}),

    html.Div([
        html.Div([
            html.Label("Select Date:", style={"font-weight": "bold", "font-size": "16px"}),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date_placeholder_text='DD.MM.YY',
                end_date_placeholder_text='DD.MM.YY',
                initial_visible_month=min_date,
                number_of_months_shown=2,
                start_date=min_date,
                end_date=(min_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)).date(),
                display_format='DD.MM.YY',
                clearable=False,
                style={'width': '400px', 'font-size': '16px', 'display': 'inline-block', 'padding': '10px', 'box-shadow': '0px 2px 5px rgba(0, 0, 0, 0.1)', 'border-radius': '5px', 'zIndex': 3, 'fontSize': '12px'}
            ),
        ], style={"width": "30%", "margin": "auto", "position": "relative", "zIndex": 3}),

        html.Div(id='date-slider-labels', style={"text-align": "center", "margin-top": "10px", "font-size": "16px"}),

        html.Div([
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
            ], style={"width": "70%", "height": "700px", "position": "relative", "zIndex": 1, "border": "2px solid #f0f0f0", "border-radius": "10px", "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.1)"}),
            
            html.Div([
                html.H4("Fire Details", style={"text-align": "center"}),
                dash_table.DataTable(
                    id='fire-details-table',
                    columns=[
                        {'name': 'Attribute', 'id': 'attribute'},
                        {'name': 'Value', 'id': 'value'}
                    ],
                    style_table={'width': '100%', 'margin': '0 auto', 'border': '1px solid #ddd', 'box-shadow': '0px 4px 10px rgba(0, 0, 0, 0.1)'},
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f9f9f9'},
                    data=[]
                )
            ], style={"width": "28%", "padding": "20px", "margin": "10px", "background-color": "#ffffff", "border-radius": "10px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.1)", "float": "right"})
        ], style={"display": "flex"}),

        html.Div(id='fire-stats', style={"text-align": "center", "margin-top": "20px", "font-size": "18px", "font-weight": "bold"})
    ], style={"margin": "20px", "padding": "20px", "background-color": "#ffffff", "border-radius": "10px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.1)"})
])

# Fire markers colored by their label
def generate_fire_markers(data):
    markers = []
    # If there are more than 200 fires, cluster them to reduce the number of markers
    if len(data) > 500:
        num_clusters = 500  # Limit the number of clusters to 500
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
                                                   iconSize=[10, 15], iconAnchor=[5, 15], popupAnchor=[1, -14], tooltipAnchor=[6, -12]),
                                         id={'type': 'fire-marker', 'index': row.name}))
        else:  # Clustered points
            lat = group['LATITUDE'].mean()
            lon = group['LONGITUDE'].mean()
            count = len(group)
            color = 'red' if group['ABNORMAL_LABEL_DECAY'].iloc[0] else 'blue'
            tooltip_text = f"Cluster of {count} fires"
            icon_size = min(20, 10 + count * 2)  # Dynamically adjust icon size based on cluster size
            markers.append(dl.Marker(position=[lat, lon], children=dl.Tooltip(tooltip_text),
                                     icon=dict(iconUrl=f'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-{color}.png',
                                               iconSize=[icon_size, icon_size + 10], iconAnchor=[icon_size // 2, icon_size + 10], popupAnchor=[1, -icon_size - 9], tooltipAnchor=[8, -14]),
                                     id={'type': 'fire-cluster', 'index': cluster_id}))
    return markers

# Update the map and statistics based on the selected date range
@app.callback(
    [Output('fire-layer', 'children'),
     Output('date-slider-labels', 'children'),
     Output('fire-stats', 'children')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('fire-map', 'zoom'),
     Input('fire-map', 'bounds')],
    prevent_initial_call=False
)
def update_fire_layer(start_date, end_date, zoom_level, bounds):
    if not bounds or len(bounds) < 2:
        # Set default bounds if they are not provided or incomplete
        bounds = [[44.0, 22.0], [52.0, 40.0]]  # Approximate bounds for Ukraine
    
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    # Ensure the selected date range is not more than 1 month apart
    if (end_date - start_date).days > 31:
        end_date = (pd.to_datetime(start_date) + pd.DateOffset(months=1) - pd.DateOffset(days=1)).date()
    
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
    stats = f"Total Abnormal Fires: {abnormal_fires}"
    
    return generate_fire_markers(filtered_data), labels, stats

# Update the table with fire details based on selected fire
@app.callback(
    Output('fire-details-table', 'data'),
    [Input({'type': 'fire-marker', 'index': dash.dependencies.ALL}, 'n_clicks'),
     Input({'type': 'fire-cluster', 'index': dash.dependencies.ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_fire_details(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return []
    else:
        trigger_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        if trigger_id['type'] == 'fire-marker':
            selected_fire = fires_gdf.iloc[trigger_id['index']]
            data = [
                {'attribute': 'Date', 'value': str(selected_fire['ACQ_DATE'])},
                {'attribute': 'Latitude', 'value': selected_fire['LATITUDE']},
                {'attribute': 'Longitude', 'value': selected_fire['LONGITUDE']},
                {'attribute': 'Significance Score', 'value': selected_fire['SIGNIFICANCE_SCORE_DECAY']},
                {'attribute': 'Abnormal Label', 'value': selected_fire['ABNORMAL_LABEL_DECAY']}
            ]
        elif trigger_id['type'] == 'fire-cluster':
            # Mock data for clusters as an example
            data = [
                {'attribute': 'Cluster Size', 'value': '15'},
                {'attribute': 'Average Significance Score', 'value': '0.85'},
                {'attribute': 'Abnormal Fires', 'value': '10'},
                {'attribute': 'Cluster Description', 'value': 'Cluster of fires in a region of Ukraine'}
            ]
        return data

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)