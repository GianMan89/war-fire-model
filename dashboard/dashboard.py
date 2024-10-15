import dash
from dash import dcc, html, dash_table, Output, Input
import dash_leaflet as dl
import pandas as pd
import geopandas as gpd
import json
import warnings
import plotly.graph_objs as go

# Suppress warnings
warnings.filterwarnings("ignore")

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
        dl.LayersControl(id='layers-control', position='topright', children=[
            dl.BaseLayer(dl.TileLayer(url='https://{s}.tile.openstreetmap.de/{z}/{x}/{y}.png',
                                     attribution='Map data © OpenStreetMap contributors', detectRetina=True),
                         name='OpenStreetMap', checked=True),
            dl.Overlay(dl.LayerGroup(id='ukraine-borders-layer', children=[
                dl.GeoJSON(data=json.loads(ukraine_borders.to_json()),
                           options=dict(style=dict(color='black', weight=3, opacity=1.0)))
            ]), name='Ukraine Borders', checked=True),
            dl.Overlay(dl.LayerGroup(id='rus-control-layer', children=[
                dl.GeoJSON(data=json.loads(rus_control.to_json()),
                           options=dict(style=dict(color='red', weight=2, fill=True, fillColor='red', fillOpacity=0.3, dashArray='5, 5')))
            ]), name='Russian-Occupied Areas', checked=True)
        ]),
        dl.LayerGroup(id='fire-layer', children=[]),
        dl.ScaleControl(position='topleft', metric=True, imperial=True)
    ], style={"width": "100vw", "height": "100vh", "position": "absolute", "top": 0, "left": 0, "zIndex": 1}),

    html.Div([
        html.Div(id='selected-date', style={"margin-bottom": "10px", "font-weight": "bold", "font-size": "16px", "color": "#003366"}),
        dcc.Graph(
            id='fires-per-day-plot',
            config={'displayModeBar': False},
            style={'height': '180px', 'margin-bottom': '0px'}
        ),
    ], style={"position": "absolute", "bottom": "10px", "left": "5%", "right": "5%", "background-color": "#f0f0f0", "padding": "20px", "border-radius": "5px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.15)", "zIndex": 2, "border": "1px solid #cccccc"}),

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
    ], style={"position": "absolute", "top": "10px", "left": "100px", "background-color": "#ffffff", "padding": "20px", "border-radius": "5px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.15)", "zIndex": 2, "display": "none", "border": "1px solid #cccccc"}, id='fire-details-container'),

    html.Div(id='layer-log')
])

# Fire markers colored by their label
def generate_fire_markers(data):
    markers = []
    for _, row in data.iterrows():
        markers.append(dl.CircleMarker(
            center=[row.geometry.y, row.geometry.x],
            radius=10,
            color='#cc0000', # if row['ABNORMAL_LABEL_DECAY'] == 1 else '#003366',
            fillColor='#cc0000',# if row['ABNORMAL_LABEL_DECAY'] == 1 else '#003366',
            fill=True,
            fillOpacity=row['SIGNIFICANCE_SCORE_DECAY'],
            opacity=0.0,
            id={'type': 'fire-marker', 'index': row.name},
            n_clicks=0,
            interactive=True  # Makes the circle marker clickable
        ))
    return markers

# Load fires only once based on the selected date range
@app.callback(
    [Output('fire-layer', 'children'),
     Output('selected-date', 'children')],
    [Input('fires-per-day-plot', 'clickData')]
)
def update_fire_layer(clickData):
    if not clickData:
        return [], "Select a date from the plot."
    
    selected_date = pd.to_datetime(clickData['points'][0]['x']).date()
    
    # Filter data based on the selected date and abnormal label
    filtered_data = fires_gdf[fires_gdf['ACQ_DATE'] == selected_date]
    selected_date_str = f"Selected Date: {selected_date.strftime('%d-%m-%Y')}"
    return generate_fire_markers(filtered_data), selected_date_str

# Log the base layer and overlay selections
@app.callback(
    Output('layer-log', 'children'),
    [Input('layers-control', 'baseLayer'), Input('layers-control', 'overlays')],
    prevent_initial_call=True
)
def log_layers(base_layer, overlays):
    return f"Base layer is {base_layer}, selected overlay(s): {json.dumps(overlays)}"

# Update the table with fire details based on marker click
@app.callback(
    [Output('fire-details-table', 'data'),
     Output('fire-details-container', 'style')],
    [Input({'type': 'fire-marker', 'index': dash.dependencies.ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_fire_details(marker_clicks):
    ctx = dash.callback_context
    if not ctx.triggered or all(click is None for click in marker_clicks):
        return [], {'display': 'none'}

    marker_id = ctx.triggered[0]['prop_id'].split('.')[0]
    index = int(json.loads(marker_id)['index'])
    row = fires_gdf.loc[index]

    data = [{
        'ACQ_DATE': str(row['ACQ_DATE']),
        'LATITUDE': row['LATITUDE'],
        'LONGITUDE': row['LONGITUDE'],
        'SIGNIFICANCE_SCORE_DECAY': f"{round(row['SIGNIFICANCE_SCORE_DECAY'] * 100, 2)}%",
        'FIRE_TYPE': "War-related" if row['ABNORMAL_LABEL_DECAY'] == 1 else "Non war-related"
    }]
    
    return data, {"position": "absolute", "top": "10px", "left": "100px", "background-color": "#ffffff", "padding": "20px", "border-radius": "5px", "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.15)", "zIndex": 2, "display": "block", "border": "1px solid #cccccc"}

# Plot the number of fire events per day
@app.callback(
    Output('fires-per-day-plot', 'figure'),
    [Input('fires-per-day-plot', 'clickData')]
)
def update_fires_per_day_plot(clickData):
    daily_fire_counts = fires_gdf['ACQ_DATE'].value_counts().sort_index()
    selected_date = pd.to_datetime(clickData['points'][0]['x']).date() if clickData else None
    selected_count = daily_fire_counts.get(selected_date, 0) if selected_date else None
    max_fire_count = daily_fire_counts.max()
    
    text_position = 'top center' if selected_count and selected_count <= 0.5 * max_fire_count else 'bottom left'
    
    figure = go.Figure(data=[
        go.Scatter(x=daily_fire_counts.index, 
                   y=daily_fire_counts.values, 
                   mode='lines+markers', 
                   line=dict(width=2, color='#003366'), 
                   hovertemplate='%{x|%b %d, %Y}, Fire Count: %{y}',
                   ),
        go.Scatter(
            x=[selected_date] if selected_date else [], y=[selected_count] if selected_count else [],
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
        plot_bgcolor='#f0f0f0'
    )
    return figure

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)