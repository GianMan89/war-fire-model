import os
import sys
import ast
import dash
from dash import dcc, html, dash_table, Output, Input, State
from dash import no_update
import dash_leaflet as dl
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import warnings
import plotly.graph_objs as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
try:
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_path, '../'))  # Adjust based on depth
    sys.path.append(project_root)
except Exception as e:
    raise RuntimeError(f"Failed to add project root to Python path: {e}")

from src.preprocessing.load_data import DataLoader

# Load data
fires_data = pd.read_csv('results/50km/test_predictions.csv')
fires_data = fires_data[fires_data['ABNORMAL_LABEL_DECAY'] == 1]

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

# Load weather data
min_date = fires_gdf['ACQ_DATE'].min()
max_date = fires_gdf['ACQ_DATE'].max()
weather_data = DataLoader.load_weather_data(min_date, max_date)
min_temp = weather_data['TEMPERATURE_2M_MEAN (°C)'].min()
max_temp = weather_data['TEMPERATURE_2M_MEAN (°C)'].max()

# Load UK MOD maps data
uk_mod_maps = pd.read_csv('data/ukr_war_events/ukr_img_overlays.csv')
# Convert date columns to datetime
uk_mod_maps['start_date'] = pd.to_datetime(uk_mod_maps['start_date']).dt.date
uk_mod_maps['end_date'] = pd.to_datetime(uk_mod_maps['end_date']).dt.date
# Parse the bounds from string to list
uk_mod_maps['bounds'] = uk_mod_maps['bounds'].apply(ast.literal_eval)

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Ukraine Forest Fires Dashboard"

# Map of Ukraine
ukraine_center = [48.3794, 31.1656]
max_bounds = [[42.0, 20.0], [54.0, 42.0]]

# Layout
app.layout = html.Div([
    dcc.Store(id='overlays-store', data=[]),
    dl.Map(id='fire-map', center=ukraine_center, zoom=6, minZoom=6, maxBounds=max_bounds, children=[
        dl.LayersControl(id='layers-control', position='topright', children=[
            dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.openstreetmap.de/{z}/{x}/{y}.png',
                                     attribution='Map data © OpenStreetMap contributors', detectRetina=True),
                         name='OpenStreetMap', checked=True),
            dl.Overlay(dl.LayerGroup(id='ukraine-borders-layer', children=[
                dl.GeoJSON(data=json.loads(ukraine_borders.to_json()),
                           options=dict(style=dict(color='black', weight=3, opacity=1.0, fillOpacity=0.0)))
            ]), name='Ukraine Borders', checked=True),
            dl.Overlay(dl.LayerGroup(id='ukraine-cloud-layer', children=[]), name='Ukraine Cloud Cover', checked=False),
            dl.Overlay(dl.LayerGroup(id='ukraine-temp-layer', children=[]), name='Ukraine Mean Temperature', checked=False),
            dl.Overlay(dl.LayerGroup(id='rus-control-layer', children=[
                dl.GeoJSON(data=json.loads(rus_control.to_json()),
                           options=dict(style=dict(color='red', weight=2, fill=True, fillColor='red', fillOpacity=0.1, dashArray='5, 5')))
            ]), name='Russian-Occupied Areas', checked=True),
            dl.Overlay(dl.LayerGroup(id='significance-opacity-layer', children=[]), name='Use Significance for Opacity', checked=False),
            dl.Overlay(
                dl.LayerGroup(id='uk-mod-map-layer', children=[]),
                name='UK MOD Map',
                checked=False
            ),
        ]),
        dl.Pane(dl.LayerGroup(id='fire-layer', children=[]), name='fire-pane', style=dict(zIndex=500)),
        dl.Pane(dl.LayerGroup(id='selected-fire-layer', children=[]), name='selected-fire-pane', style=dict(zIndex=501)),
        dl.Pane(dl.LayerGroup(id='fire-tooltip-layer', children=[]), name='fire-tooltip-pane', style=dict(zIndex=502)),
        dl.ScaleControl(position='topleft', metric=True, imperial=True),
    ], style={"width": "100vw", "height": "100vh", "position": "absolute", "top": 0, "left": 0, "zIndex": 1}),

    html.Div([
    # Container for selected date and slider
        html.Div([
            # Selected date text on the left
            html.Div(id='selected-date', style={
                "font-weight": "bold",
                "font-size": "16px",
                "color": "#003366",
                'font-family': 'Arial',
                "display": "inline-block",
                "verticalAlign": "middle",
                "width": "50%"  # Adjust the width as needed
            }),

            # Slider on the right
            html.Div([
                html.Label('Number of Clusters:', style={
                    "font-weight": "bold",
                    "font-size": "16px",
                    "color": "#003366",
                    'font-family': 'Arial',
                    "margin-right": "10px"
                }),
                dcc.Input(
                    id='n-clusters-input',
                    type='number',
                    min=1,
                    max=500,  # Adjust max as needed
                    value=500,  # Default value
                    debounce=True, # Debounce input to prevent rapid updates
                    style={
                        "width": "60px",
                        "display": "inline-block",
                        "verticalAlign": "middle",
                        "font-family": "Arial",
                        "font-size": "16px",
                        "margin-bottom": "5px"
                    }
                )
            ], style={
                "display": "inline-block",
                "verticalAlign": "middle",
                "width": "50%",
                "textAlign": "right"
            }),
        ], style={
            "width": "100%",
            "display": "flex",
            "justify-content": "space-between",
            "alignItems": "center",
            "margin-bottom": "10px"
        }),

        # The fires per day plot
        dcc.Graph(
            id='fires-per-day-plot',
            config={'displayModeBar': False},
            style={
                'height': '180px',
                'margin-bottom': '0px',
                'box-shadow': '0px 4px 10px rgba(0, 0, 0, 0.15)',
                'border': '1px solid #003366'
            }
        ),
    ], style={
        "position": "absolute",
        "bottom": "10px",
        "left": "5%",
        "right": "5%",
        "background-color": "#e6e6e6",
        "padding": "20px",
        "border-radius": "5px",
        "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.15)",
        "zIndex": 2,
        "border": "1px solid #cccccc"
    }),

    html.Div([
        dash_table.DataTable(
            id='fire-details-table',
            columns=[
                {'name': 'Date', 'id': 'ACQ_DATE'},
                {'name': 'Latitude', 'id': 'LATITUDE'},
                {'name': 'Longitude', 'id': 'LONGITUDE'},
                {'name': 'Significance', 'id': 'SIGNIFICANCE_SCORE_DECAY'},
                {'name': 'Fire Type', 'id': 'FIRE_TYPE'},
                {'name': 'Cluster Size', 'id': 'CLUSTER_SIZE'}
            ],
            style_table={'width': '100%', 'margin': '0 auto', 'border': '1px solid #003366', 'box-shadow': '0px 4px 10px rgba(0, 0, 0, 0.15)'},
            style_cell={'textAlign': 'center', 'padding': '10px', 'font-family': 'Arial', 'font-size': '14px', 'color': '#003366'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#e6e6e6'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
            style_as_list_view=True,
            data=[],
        )
    ], style={"position": "absolute", 
              "top": "10px", 
              "left": "120px", 
              "background-color": "#ffffff", 
              "padding": "20px", 
              "border-radius": "5px", 
              "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.15)", 
              "zIndex": 2, 
              "display": "none", 
              "border": "1px solid #cccccc"}, 
              id='fire-details-container'),
])

def get_marker_size(num_fires):
    # Adjust this function to scale marker sizes appropriately
    base_size = 5
    size = base_size + np.log1p(num_fires) * 2  # Logarithmic scaling
    return size

def generate_fire_markers_without_clustering(data, use_significance_opacity):
    markers = []
    for _, row in data.iterrows():
        markers.append(dl.CircleMarker(
            center=[row.geometry.y, row.geometry.x],
            radius=8,
            color='#cc0000',
            fillColor='#cc0000',
            fill=True,
            fillOpacity=row['SIGNIFICANCE_SCORE_DECAY'] if use_significance_opacity else 0.5,
            opacity=0.0 if use_significance_opacity else 1.0,
            id={'type': 'fire-marker', 'index': row.name, 'significance_opt': use_significance_opacity},
            n_clicks=0,
            interactive=True,
            children=[dl.Tooltip(
                content=f"Single fire<br>"
                        f"Date: {row['ACQ_DATE']}<br>"
                        f"Lat: {round(row['LATITUDE'], 4)}<br>"
                        f"Lon: {round(row['LONGITUDE'], 4)}<br>"
                        f"Significance: {round(row['SIGNIFICANCE_SCORE_DECAY'] * 100, 2)}%",
                direction='auto', permanent=False, sticky=False, interactive=True, offset=[0, 0], opacity=0.9,
                pane='fire-tooltip-pane',
            )]
        ))
    return markers

# Fire markers colored by their label
def generate_fire_markers(data, use_significance_opacity, n_clusters):
    # If the number of fires is less than or equal to n_clusters, skip clustering
    if len(data) <= n_clusters or n_clusters <= 1:
        return generate_fire_markers_without_clustering(data, use_significance_opacity)
    
    # Prepare data for clustering
    coords = np.array([[row.geometry.y, row.geometry.x] for idx, row in data.iterrows()])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(coords)
    data['cluster'] = kmeans.labels_
    
    # Group data by clusters
    cluster_groups = data.groupby('cluster')
    
    markers = []
    for cluster_label, cluster_data in cluster_groups:
        # Cluster center coordinates
        cluster_center_lat = cluster_data.geometry.y.mean()
        cluster_center_lon = cluster_data.geometry.x.mean()
        
        # Mean significance score
        mean_significance = cluster_data['SIGNIFICANCE_SCORE_DECAY'].mean()
        
        # Number of fires in the cluster
        num_fires = len(cluster_data)
        
        # Determine marker size based on the number of fires
        marker_size = get_marker_size(num_fires)
        
        # Create cluster marker
        marker = dl.CircleMarker(
            center=[cluster_center_lat, cluster_center_lon],
            radius=marker_size,
            color='#cc0000',
            fillColor='#cc0000',
            fill=True,
            fillOpacity=mean_significance if use_significance_opacity else 0.5,
            opacity=0.0 if use_significance_opacity else 1.0,
            id={'type': 'fire-cluster-marker', 'index': int(cluster_label), 'significance_opt': use_significance_opacity},
            n_clicks=0,
            interactive=True,
            children=[dl.Tooltip(
                content=(
                    [f"Cluster of {num_fires} fires<br>" if num_fires > 1 else "Single fire<br>"][0] +
                    f"Date: {cluster_data['ACQ_DATE'].iloc[0]}<br>"
                    f"Lat: {round(cluster_center_lat, 4)}<br>"
                    f"Lon: {round(cluster_center_lon, 4)}<br>"
                    f"{["Mean Significance" if num_fires > 1 else "Significance"][0]}: {round(mean_significance * 100, 2)}%"
                ),
                direction='auto',
                permanent=False,
                sticky=False,
                interactive=True,
                offset=[0, 0],
                opacity=0.9,
                pane='fire-tooltip-pane',
            )]
        )
        markers.append(marker)
    return markers



# Function to get cloud cover opacity for a specific oblast and date
def get_cloud_cover_opacity(oblast_id, acq_date):
    cloud_cover = weather_data[(weather_data['OBLAST_ID'] == oblast_id) & (weather_data['ACQ_DATE'] == acq_date)]['CLOUD_COVER (%)'].values
    if len(cloud_cover) > 0:
        return cloud_cover[0] / 100
    return 0  # Default zero opacity if no data is found

# Generate Ukraine borders with dynamic cloud cover opacity
def generate_ukraine_cloud_layer(selected_date):
    layers = [
        dl.GeoJSON(
            id=f'cloud-geojson-{i}-{selected_date}',
            data=json.loads(ukraine_borders.iloc[i:i+1].to_json()),
            options=dict(style=dict(color='black', weight=3, opacity=1.0, fillColor='darkgrey',
                                    fillOpacity=get_cloud_cover_opacity(ukraine_borders.iloc[i]['id'], selected_date))),
            children=[dl.Tooltip(content=f"Cloud Cover: {round(get_cloud_cover_opacity(ukraine_borders.iloc[i]['id'], selected_date) * 100, 2)}%", 
                                    direction='auto', permanent=False, sticky=True, interactive=True, offset=[0, 0], opacity=0.9,
                                    id=f'cloud-tooltip-{i}-{selected_date}')]
        ) for i in range(len(ukraine_borders))
    ]
    return layers

# Function to get mean temperature for a specific oblast and date
def get_mean_temperature(oblast_id, acq_date):
    temperature = weather_data[(weather_data['OBLAST_ID'] == oblast_id) & (weather_data['ACQ_DATE'] == acq_date)]['TEMPERATURE_2M_MEAN (°C)'].values
    if len(temperature) > 0:
        return temperature[0]
    return 0  # Default zero temperature if no data is found

# Generate Ukraine temperature layer with color scale based on temperature
def generate_ukraine_temp_layer(selected_date):
    temperatures = [get_mean_temperature(ukraine_borders.iloc[i]['id'], selected_date) for i in range(len(ukraine_borders))]
    norm = mcolors.Normalize(vmin=min_temp, vmax=max_temp)
    cmap = cm.get_cmap('coolwarm')

    layers = [
        dl.GeoJSON(
            id=f'temp-geojson-{i}-{selected_date}',
            data=json.loads(ukraine_borders.iloc[i:i+1].to_json()),
            options=dict(style=dict(fillColor=mcolors.to_hex(cmap(norm(temperatures[i]))), 
                                    color='black', weight=3, opacity=1.0, fillOpacity=0.8)),
            children=[dl.Tooltip(content=f"Temperature: {round(temperatures[i], 2)}°C", direction='auto', 
                                 permanent=False, sticky=True, interactive=True, offset=[0, 0], opacity=0.9,
                                 id=f'temp-tooltip-{i}-{selected_date}')]
        ) for i in range(len(ukraine_borders))
    ]
    return layers

# Load fires and update layers based on the selected date
@app.callback(
    [
        Output('fire-layer', 'children'),
        Output('selected-date', 'children'),
        Output('ukraine-cloud-layer', 'children'),
        Output('ukraine-temp-layer', 'children'),
        Output('uk-mod-map-layer', 'children'),  # New output
        Output('overlays-store', 'data')
    ],
    [
        Input('fires-per-day-plot', 'clickData'),
        Input('layers-control', 'overlays'),
        Input('n-clusters-input', 'value')
    ],
    [
        State('overlays-store', 'data')
    ]
)
def update_layers(clickData, overlays, n_clusters, prev_overlays):
    # Handle n_clusters being None or invalid
    if n_clusters is None or n_clusters < 1:
        n_clusters = 10  # Default value

    # If no date is selected, do not update the map and print a message
    if not clickData:
        return dash.no_update, "Select a date from the plot.", dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Initialize outputs
    fire_markers = dash.no_update
    selected_date_str = dash.no_update
    ukraine_cloud_layer = dash.no_update
    ukraine_temp_layer = dash.no_update
    uk_mod_map_layer = dash.no_update

    # Initialize prev_overlays if None
    if prev_overlays is None:
        prev_overlays = []

    # Determine which Input triggered the callback
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]
    # Convert overlays lists to sets for comparison
    current_overlays_set = set(overlays) if overlays else set()
    prev_overlays_set = set(prev_overlays) if prev_overlays else set()
    # Determine overlays that were added or removed
    added_overlays = current_overlays_set - prev_overlays_set
    removed_overlays = prev_overlays_set - current_overlays_set
    # Determine if 'Use Significance for Opacity' was toggled
    significance_toggled = 'Use Significance for Opacity' in added_overlays or 'Use Significance for Opacity' in removed_overlays
    # Determine if other overlays were toggled
    cloud_layer_toggled = 'Ukraine Cloud Cover' in added_overlays or 'Ukraine Cloud Cover' in removed_overlays
    temp_layer_toggled = 'Ukraine Mean Temperature' in added_overlays or 'Ukraine Mean Temperature' in removed_overlays
    # Determine if 'UK MOD Map' overlay is toggled
    uk_mod_map_toggled = 'UK MOD Map' in added_overlays or 'UK MOD Map' in removed_overlays

    # Handle clickData changes
    if triggered_input == 'fires-per-day-plot':
        if not clickData:
            return [], "Select a date from the plot.", [], [], overlays
        selected_date = pd.to_datetime(clickData['points'][0]['x']).date()
        selected_date_str = f"Selected Date: {selected_date.strftime('%d-%m-%Y')}, Number of Abnormal Fires: {len(fires_gdf[fires_gdf['ACQ_DATE'] == selected_date])}"
        # Filter data based on the selected date
        filtered_data = fires_gdf[fires_gdf['ACQ_DATE'] == selected_date]
        # Determine if 'Use Significance for Opacity' is in overlays
        use_significance_opacity = 'Use Significance for Opacity' in overlays
        # Generate fire markers
        fire_markers = generate_fire_markers(filtered_data, use_significance_opacity, n_clusters)
        # Update other layers
        ukraine_cloud_layer = generate_ukraine_cloud_layer(selected_date)
        ukraine_temp_layer = generate_ukraine_temp_layer(selected_date)
        # Update UK MOD Map layer if it's enabled
        if 'UK MOD Map' in overlays:
            map_row = uk_mod_maps[(uk_mod_maps['start_date'] <= selected_date) & (uk_mod_maps['end_date'] >= selected_date)]
            if not map_row.empty:
                map_info = map_row.iloc[0]
                image_overlay = dl.ImageOverlay(
                    url=map_info['url'],
                    bounds=map_info['bounds'],
                    opacity=0.7,
                    id='uk-mod-image-overlay'
                )
                uk_mod_map_layer = [image_overlay]
            else:
                uk_mod_map_layer = []
        else:
            uk_mod_map_layer = []
    elif triggered_input == 'layers-control':
        # Handle 'Use Significance for Opacity' toggle
        if significance_toggled:
            if clickData:
                selected_date = pd.to_datetime(clickData['points'][0]['x']).date()
                # Filter data based on the selected date
                filtered_data = fires_gdf[fires_gdf['ACQ_DATE'] == selected_date]
                # Determine if 'Use Significance for Opacity' is in overlays
                use_significance_opacity = 'Use Significance for Opacity' in overlays
                # Generate fire markers
                fire_markers = generate_fire_markers(filtered_data, use_significance_opacity, n_clusters)
            else:
                # No date selected, cannot update fire markers
                pass
        # Handle 'Ukraine Cloud Cover' toggle
        if cloud_layer_toggled:
            if clickData:
                selected_date = pd.to_datetime(clickData['points'][0]['x']).date()
                # Update cloud layer
                ukraine_cloud_layer = generate_ukraine_cloud_layer(selected_date)
        # Handle 'Ukraine Mean Temperature' toggle
        if temp_layer_toggled:
            if clickData:
                selected_date = pd.to_datetime(clickData['points'][0]['x']).date()
                # Update temperature layer
                ukraine_temp_layer = generate_ukraine_temp_layer(selected_date)
        if uk_mod_map_toggled:
            if 'UK MOD Map' in overlays and clickData:
                selected_date = pd.to_datetime(clickData['points'][0]['x']).date()
                map_row = uk_mod_maps[(uk_mod_maps['start_date'] <= selected_date) & (uk_mod_maps['end_date'] >= selected_date)]
                if not map_row.empty:
                    map_info = map_row.iloc[0]
                    image_overlay = dl.ImageOverlay(
                        url=map_info['url'],
                        bounds=map_info['bounds'],
                        opacity=0.7,
                        id='uk-mod-image-overlay'
                    )
                    uk_mod_map_layer = [image_overlay]
                else:
                    uk_mod_map_layer = []
            else:
                uk_mod_map_layer = []
    elif triggered_input == 'n-clusters-input':
    # Handle changes to the cluster size input
        if clickData:
            selected_date = pd.to_datetime(clickData['points'][0]['x']).date()
            # Update the selected date string
            num_fires = len(fires_gdf[fires_gdf['ACQ_DATE'] == selected_date])
            selected_date_str = f"Selected Date: {selected_date.strftime('%d-%m-%Y')}, Number of Abnormal Fires: {num_fires}"
            # Filter data based on the selected date
            filtered_data = fires_gdf[fires_gdf['ACQ_DATE'] == selected_date]
            # Determine if 'Use Significance for Opacity' is in overlays
            use_significance_opacity = 'Use Significance for Opacity' in overlays
            # Generate fire markers with the new cluster size
            fire_markers = generate_fire_markers(filtered_data, use_significance_opacity, n_clusters)
    else:
        pass  # Other triggers

    # Handle 'UK MOD Map' overlay
    if 'UK MOD Map' in overlays:
        if clickData:
            selected_date = pd.to_datetime(clickData['points'][0]['x']).date()
            # Find the map that corresponds to the selected date
            map_row = uk_mod_maps[(uk_mod_maps['start_date'] <= selected_date) & (uk_mod_maps['end_date'] >= selected_date)]
            if not map_row.empty:
                # Get the first matching map (assuming no overlaps)
                map_info = map_row.iloc[0]
                # Create the ImageOverlay
                image_overlay = dl.ImageOverlay(
                    url=map_info['url'],
                    bounds=map_info['bounds'],
                    opacity=0.7,
                    id='uk-mod-image-overlay'
                )
                uk_mod_map_layer = [image_overlay]
            else:
                # No map for the selected date
                uk_mod_map_layer = []
        else:
            # No date selected
            uk_mod_map_layer = []
    else:
        # Overlay is not checked, clear the layer
        uk_mod_map_layer = []

    # Update the stored overlays
    return fire_markers, selected_date_str, ukraine_cloud_layer, ukraine_temp_layer, uk_mod_map_layer, overlays

# Update the table with fire details based on marker click, and mark the selected fire on the map
@app.callback(
    [
        Output('fire-details-table', 'data'),
        Output('fire-details-container', 'style'),
        Output('selected-fire-layer', 'children')
    ],
    [
        Input({'type': 'fire-marker', 'index': dash.dependencies.ALL, 'significance_opt': dash.dependencies.ALL}, 'n_clicks'),
        Input({'type': 'fire-cluster-marker', 'index': dash.dependencies.ALL, 'significance_opt': dash.dependencies.ALL}, 'n_clicks'),
        Input('fires-per-day-plot', 'clickData'),
        Input('n-clusters-input', 'value')
    ],
    [
        State('fire-details-table', 'data'),
        State('fire-details-container', 'style'),
        State('selected-fire-layer', 'children'),
    ],
    prevent_initial_call=True
)
def update_fire_details(marker_clicks, cluster_marker_clicks, clickData, n_clusters,
                        current_data, current_style, current_selected_fire_marker):
    ctx = dash.callback_context

    if not ctx.triggered:
        return no_update, no_update, no_update

    triggered_prop_id = ctx.triggered[0]['prop_id']
    triggered_value = ctx.triggered[0]['value']

    if triggered_prop_id == 'fires-per-day-plot.clickData':
        # Reset outputs when date changes
        return [], {'display': 'none'}, []
    
    elif triggered_prop_id == 'n-clusters-input.value':
        # Reset outputs when cluster size changes
        return [], {'display': 'none'}, []

    elif triggered_prop_id.endswith('.n_clicks') and triggered_value and triggered_value > 0:
        marker_id_json = triggered_prop_id.split('.')[0]
        marker_id = json.loads(marker_id_json)
        index = marker_id['index']
        marker_type = marker_id['type']

        # Get the selected date
        selected_date = pd.to_datetime(clickData['points'][0]['x']).date()
        filtered_data = fires_gdf[fires_gdf['ACQ_DATE'] == selected_date]

        if marker_type == 'fire-cluster-marker':
            # Recompute clustering to get cluster data
            coords = np.array([[row.geometry.y, row.geometry.x] for idx, row in filtered_data.iterrows()])
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            kmeans.fit(coords)
            filtered_data['cluster'] = kmeans.labels_
            cluster_data = filtered_data[filtered_data['cluster'] == index]

            # Prepare data
            data = [{
                'ACQ_DATE': str(selected_date),
                'LATITUDE': round(cluster_data.geometry.y.mean(), 4),
                'LONGITUDE': round(cluster_data.geometry.x.mean(), 4),
                'SIGNIFICANCE_SCORE_DECAY': f"{round(cluster_data['SIGNIFICANCE_SCORE_DECAY'].mean() * 100, 2)}%",
                'FIRE_TYPE': "War-related", # if row['ABNORMAL_LABEL_DECAY'] == 1 else "Non war-related",
                'CLUSTER_SIZE': f"Cluster of {len(cluster_data)} fires" if len(cluster_data) > 1 else "Single fire"
            }]

            # Create selected cluster marker
            selected_fire_marker = dl.CircleMarker(
                center=[cluster_data.geometry.y.mean(), cluster_data.geometry.x.mean()],
                radius=get_marker_size(len(cluster_data)),
                color='#6e57ce',
                fillColor='#6e57ce',
                fill=True,
                fillOpacity=0.8,
                opacity=1.0,
                id='selected-fire-marker',
                children=[
                    dl.Tooltip(
                        content=(
                            [f"Cluster of {len(cluster_data)} fires<br>" if len(cluster_data) > 1 else "Single fire<br>"][0] +
                            f"Date: {data[0]['ACQ_DATE']}<br>" +
                            f"Lat: {round(cluster_data.geometry.y.mean(), 4)}<br>" +
                            f"Lon: {round(cluster_data.geometry.x.mean(), 4)}<br>" +
                            [f"{["Mean Significance" if len(cluster_data) > 1 else "Significance"][0]}: {round(cluster_data['SIGNIFICANCE_SCORE_DECAY'].mean() * 100, 2)}%"][0]
                        ),
                        direction='auto',
                        permanent=False,
                        sticky=False,
                        interactive=True,
                        offset=[0, 0],
                        opacity=0.9,
                        pane='fire-tooltip-pane',
                        id=f'selected-fire-tooltip-{marker_id_json}'
                    )
                ]
            )
        else:
            # Handle individual fire marker clicks
            row = fires_gdf.loc[index]
            data = [{
                'ACQ_DATE': str(row['ACQ_DATE']),
                'LATITUDE': round(row['LATITUDE'], 4),
                'LONGITUDE': round(row['LONGITUDE'], 4),
                'SIGNIFICANCE_SCORE_DECAY': f"{round(row['SIGNIFICANCE_SCORE_DECAY'] * 100, 2)}%",
                'FIRE_TYPE': "War-related" if row['ABNORMAL_LABEL_DECAY'] == 1 else "Non war-related",
                'CLUSTER_SIZE': "Single fire"
            }]
            selected_fire_marker = dl.CircleMarker(
                center=[row.geometry.y, row.geometry.x],
                radius=10,
                color='#6e57ce',
                fillColor='#6e57ce',
                fill=True,
                fillOpacity=0.8,
                opacity=1.0,
                id='selected-fire-marker',
                children=[
                    dl.Tooltip(
                        content=(
                            f"Single fire<br>"
                            f"Date: {row['ACQ_DATE']}<br>"
                            f"Lat: {round(row['LATITUDE'], 4)}<br>"
                            f"Lon: {round(row['LONGITUDE'], 4)}<br>"
                            f"Significance: {round(row['SIGNIFICANCE_SCORE_DECAY'] * 100, 2)}%"
                        ),
                        direction='auto',
                        permanent=False,
                        sticky=False,
                        interactive=True,
                        offset=[0, 0],
                        opacity=0.9,
                        pane='fire-tooltip-pane',
                        id=f'selected-fire-tooltip-{marker_id_json}'
                    )
                ]
            )

        style = {
            "position": "absolute",
            "top": "10px",
            "left": "120px",
            "background-color": "#ffffff",
            "padding": "20px",
            "border-radius": "5px",
            "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.15)",
            "zIndex": 2,
            "display": "block",
            "border": "1px solid #cccccc"
        }

        return data, style, [selected_fire_marker]

    else:
        return no_update, no_update, no_update


# Plot the number of fire events per day
@app.callback(
    Output('fires-per-day-plot', 'figure'),
    [Input('fires-per-day-plot', 'clickData')]
)
def update_fires_per_day_plot(clickData):
    daily_fire_counts = fires_gdf['ACQ_DATE'].value_counts().sort_index()
    # Ensure all dates within the range have a count, filling missing dates with zero
    all_dates = pd.date_range(start=min_date, end=max_date)
    daily_fire_counts = daily_fire_counts.reindex(all_dates, fill_value=0)
    daily_fire_counts.index = daily_fire_counts.index.date
    selected_date = pd.to_datetime(clickData['points'][0]['x']).date() if clickData else None
    selected_count = daily_fire_counts.get(selected_date, 0) if selected_date else None
        
    figure = go.Figure(data=[
        go.Scatter(x=daily_fire_counts.index, 
                   y=daily_fire_counts.values, 
                   mode='lines+markers', 
                   line=dict(width=2, color='#003366'), 
                   hovertemplate='%{x|%b %d, %Y}, Fire Count: %{y}',
                   name=''
                   ),
        go.Scatter(
            x=[selected_date] if selected_date else [], 
            y=[selected_count],
            mode='markers+text',
            marker=dict(size=10, color='#cc0000'),
            hoverinfo='skip',
        )
    ])
    figure.update_layout(
        yaxis_title='Number of Fires',
        margin=dict(l=40, r=40, t=20, b=0),
        height=180,
        showlegend=False,
        plot_bgcolor='#e6e6e6'
    )
    return figure

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)