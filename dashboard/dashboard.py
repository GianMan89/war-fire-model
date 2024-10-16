import os
import sys
import dash
from dash import dcc, html, dash_table, Output, Input, State
import dash_leaflet as dl
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

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Ukraine Forest Fires Dashboard"

# Map of Ukraine
ukraine_center = [48.3794, 31.1656]

# Layout
app.layout = html.Div([
    dcc.Store(id='overlays-store', data=[]),
    dl.Map(id='fire-map', center=ukraine_center, zoom=6, children=[
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
            dl.Overlay(dl.LayerGroup(id='significance-opacity-layer', children=[]), name='Use Significance for Opacity', checked=False)
        ]),
        dl.Pane(dl.LayerGroup(id='fire-layer', children=[]), name='fire-pane', style=dict(zIndex=500)),
        dl.Pane(dl.LayerGroup(id='selected-fire-layer', children=[]), name='selected-fire-pane', style=dict(zIndex=501)),
        dl.Pane(dl.LayerGroup(id='fire-tooltip-layer', children=[]), name='fire-tooltip-pane', style=dict(zIndex=502)),
        dl.ScaleControl(position='topleft', metric=True, imperial=True)
    ], style={"width": "100vw", "height": "100vh", "position": "absolute", "top": 0, "left": 0, "zIndex": 1}),

    html.Div([
        html.Div(id='selected-date', style={"margin-bottom": "10px", "font-weight": "bold", "font-size": "16px", "color": "#003366", 'font-family': 'Arial'}),
        dcc.Graph(
            id='fires-per-day-plot',
            config={'displayModeBar': False},
            style={'height': '180px', 'margin-bottom': '0px', 'box-shadow': '0px 4px 10px rgba(0, 0, 0, 0.15)', 'border': '1px solid #003366'}
        ),
    ], style={"position": "absolute", "bottom": "10px", "left": "5%", "right": "5%", "background-color": "#e6e6e6", "padding": "20px", "border-radius": "5px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.15)", "zIndex": 2, "border": "1px solid #cccccc"}),

    html.Div([
        dash_table.DataTable(
            id='fire-details-table',
            columns=[
                {'name': 'Date', 'id': 'ACQ_DATE'},
                {'name': 'Latitude', 'id': 'LATITUDE'},
                {'name': 'Longitude', 'id': 'LONGITUDE'},
                {'name': 'Significance', 'id': 'SIGNIFICANCE_SCORE_DECAY'},
                {'name': 'Fire Type', 'id': 'FIRE_TYPE'}
            ],
            style_table={'width': '100%', 'margin': '0 auto', 'border': '1px solid #003366', 'box-shadow': '0px 4px 10px rgba(0, 0, 0, 0.15)'},
            style_cell={'textAlign': 'center', 'padding': '10px', 'font-family': 'Arial', 'font-size': '14px', 'color': '#003366'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#e6e6e6'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
            style_as_list_view=True,
            data=[],
        )
    ], style={"position": "absolute", "top": "10px", "left": "120px", "background-color": "#ffffff", "padding": "20px", "border-radius": "5px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.15)", "zIndex": 2, "display": "none", "border": "1px solid #cccccc"}, id='fire-details-container'),

    html.Div(id='layer-log')
])

# Fire markers colored by their label
def generate_fire_markers(data, use_significance_opacity):
    markers = []
    for _, row in data.iterrows():
        if use_significance_opacity:
            markers.append(dl.CircleMarker(
                center=[row.geometry.y, row.geometry.x],
                radius=8,
                color='#cc0000',
                fillColor='#cc0000',
                fill=True,
                fillOpacity=row['SIGNIFICANCE_SCORE_DECAY'],
                opacity=0.0,
                id={'type': 'fire-marker-significance', 'index': row.name},
                n_clicks=0,
                interactive=True,
                children=[dl.Tooltip(
                    content=f"Date: {row['ACQ_DATE']}<br>Lat: {row['LATITUDE']}<br>Lon: {row['LONGITUDE']}<br>Significance: {round(row['SIGNIFICANCE_SCORE_DECAY'] * 100, 2)}%", 
                    direction='auto', permanent=False, sticky=False, interactive=True, offset=[0, 0], opacity=0.9,
                    pane='fire-tooltip-pane',
                    )]
            ))
        else:
            markers.append(dl.CircleMarker(
                center=[row.geometry.y, row.geometry.x],
                radius=8,
                color='#cc0000',
                fillColor='#cc0000',
                fill=True,
                fillOpacity=0.5,
                opacity=1.0,
                id={'type': 'fire-marker', 'index': row.name},
                n_clicks=0,
                interactive=True,
                children=[dl.Tooltip(
                    content=f"Date: {row['ACQ_DATE']}<br>Lat: {row['LATITUDE']}<br>Lon: {row['LONGITUDE']}<br>Significance: {round(row['SIGNIFICANCE_SCORE_DECAY'] * 100, 2)}%",
                    direction='auto', permanent=False, sticky=False, interactive=True, offset=[0, 0], opacity=0.9,
                    pane='fire-tooltip-pane',
                    )]
            ))
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
        Output('overlays-store', 'data')  # Update the stored overlays
    ],
    [
        Input('fires-per-day-plot', 'clickData'),
        Input('layers-control', 'overlays')
    ],
    [
        State('overlays-store', 'data')
    ]
)
def update_layers(clickData, overlays, prev_overlays):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Initialize outputs
    fire_markers = dash.no_update
    selected_date_str = dash.no_update
    ukraine_cloud_layer = dash.no_update
    ukraine_temp_layer = dash.no_update

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

    # Handle clickData changes
    if triggered_input == 'fires-per-day-plot':
        if not clickData:
            return [], "Select a date from the plot.", [], [], overlays
        selected_date = pd.to_datetime(clickData['points'][0]['x']).date()
        selected_date_str = f"Selected Date: {selected_date.strftime('%d-%m-%Y')}"
        # Filter data based on the selected date
        filtered_data = fires_gdf[fires_gdf['ACQ_DATE'] == selected_date]
        # Determine if 'Use Significance for Opacity' is in overlays
        use_significance_opacity = 'Use Significance for Opacity' in overlays
        # Generate fire markers
        fire_markers = generate_fire_markers(filtered_data, use_significance_opacity)
        # Update other layers
        ukraine_cloud_layer = generate_ukraine_cloud_layer(selected_date)
        ukraine_temp_layer = generate_ukraine_temp_layer(selected_date)
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
                fire_markers = generate_fire_markers(filtered_data, use_significance_opacity)
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
    else:
        pass  # Other triggers

    # Update the stored overlays
    return fire_markers, selected_date_str, ukraine_cloud_layer, ukraine_temp_layer, overlays

# Update the table with fire details based on marker click, and mark the selected fire on the map
@app.callback(
    [Output('fire-details-table', 'data'),
     Output('fire-details-container', 'style'),
     Output('selected-fire-layer', 'children')],
    [Input({'type': 'fire-marker-significance', 'index': dash.dependencies.ALL}, 'n_clicks'),
     Input({'type': 'fire-marker', 'index': dash.dependencies.ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_fire_details(marker_clicks_significance, marker_clicks):
    ctx = dash.callback_context
    if not ctx.triggered or (all(click is None for click in marker_clicks) and all(click is None for click in marker_clicks_significance)):
        return [], {'display': 'none'}, []

    # Extract the triggering property ID and value
    triggered_prop_id = ctx.triggered[0]['prop_id']
    triggered_value = ctx.triggered[0]['value']

    # If the triggering value is None or 0, no actual click has occurred
    if not triggered_value or triggered_value == 0:
        return [], {'display': 'none'}, []

    marker_id = triggered_prop_id.split('.')[0]
    index = int(json.loads(marker_id)['index'])
    row = fires_gdf.loc[index]

    data = [{
        'ACQ_DATE': str(row['ACQ_DATE']),
        'LATITUDE': row['LATITUDE'],
        'LONGITUDE': row['LONGITUDE'],
        'SIGNIFICANCE_SCORE_DECAY': f"{round(row['SIGNIFICANCE_SCORE_DECAY'] * 100, 2)}%",
        'FIRE_TYPE': "War-related" if row['ABNORMAL_LABEL_DECAY'] == 1 else "Non war-related"
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
        children=[dl.Tooltip(
                    content=f"Date: {row['ACQ_DATE']}<br>Lat: {row['LATITUDE']}<br>Lon: {row['LONGITUDE']}<br>Significance: {round(row['SIGNIFICANCE_SCORE_DECAY'] * 100, 2)}%",
                    direction='auto', permanent=False, sticky=False, interactive=True, offset=[0, 0], opacity=0.9,
                    pane='fire-tooltip-pane', id=f'selected-fire-tooltip-{marker_id}'
                    )]
    )
    
    return data, {"position": "absolute", "top": "10px", "left": "120px", "background-color": "#ffffff", "padding": "20px", "border-radius": "5px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.15)", "zIndex": 2, "display": "block", "border": "1px solid #cccccc"}, [selected_fire_marker]

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
    max_fire_count = daily_fire_counts.max()
    
    text_position = 'top center' if (selected_count and selected_count <= 0.5 * max_fire_count) or selected_count == 0 else 'bottom left'
    
    figure = go.Figure(data=[
        go.Scatter(x=daily_fire_counts.index, 
                   y=daily_fire_counts.values, 
                   mode='lines+markers', 
                   line=dict(width=2, color='#003366'), 
                   hovertemplate='%{x|%b %d, %Y}, Fire Count: %{y}',
                   ),
        go.Scatter(
            x=[selected_date] if selected_date else [], 
            y=[selected_count],# if selected_count else [],
            mode='markers+text',
            marker=dict(size=10, color='#cc0000'),
            text=[f'{selected_count} fires<br>'],
            textposition=text_position,
            textfont=dict(family='Arial', size=14, color='black'),
            texttemplate='<b>%{text}</b>',
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