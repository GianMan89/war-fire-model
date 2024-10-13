import os
import sys
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
import warnings

warnings.filterwarnings("ignore")

# Add the project root to the Python path
try:
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
    sys.path.append(project_root)
except Exception as e:
    raise RuntimeError(f"Failed to add project root to Python path: {e}")

try:
    from utils.file_utils import get_path
    from config.config import get_parameter
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    sys.exit(1)


class UkraineFirePlotting:
    """
    A class for plotting fire data and events in Ukraine.

    Attributes
    -------
    major_cities : dict
        A dictionary containing major cities in Ukraine with their coordinates and text shifts for plotting.
    war_events : pd.DataFrame
        A DataFrame containing war events in Ukraine with start and end dates.
    ukraine_borders : gpd.GeoDataFrame
        A GeoDataFrame containing the shapefile of Ukrainian borders.
    rus_adv : gpd.GeoDataFrame
        A GeoDataFrame containing the shapefile of Russian-occupied territories in Ukraine.
    
    Methods
    -------
    plot_fire_data(data, event='', start_date=None, end_date=None, colors=['red'], alphas=[0.25], legend=False,
                     markersizes=[1], plot_cities=False, city_color='yellow', cmaps=None, column=None)
          Plot fire data within Ukrainian borders.
    plot_histogram(normal_scores, abnormal_scores, start_date, end_date, event, column)
            Plot histograms for significance scores of normal and abnormal fires.
    """
    def __init__(self):
        try:
            # Load major cities in Ukraine with their coordinates and text shifts for plotting
            major_cities_df = pd.read_csv(f'{get_path("oblasts_data_dir")}/ukr_cities.csv')
            self.major_cities = {
                row['CITY']: (row['LATITUDE'], row['LONGITUDE'], row['LAT_TXT_SHIFT'], row['LON_TXT_SHIFT']) 
                for _, row in major_cities_df.iterrows()
            }
        except FileNotFoundError:
            print("Error: Major cities data file not found.")
            self.major_cities = {}
        except Exception as e:
            print(f"Unexpected error while loading major cities data: {e}")
            self.major_cities = {}

        try:
            # Load war events in Ukraine with start and end dates, and convert to datetime.date
            war_events_df = pd.read_csv(f'{get_path("war_events_data_dir")}/ukr_war_events.csv')
            war_events_df['START_DATE'] = pd.to_datetime(war_events_df['START_DATE']).dt.date
            war_events_df['END_DATE'] = pd.to_datetime(war_events_df['END_DATE']).dt.date
            self.war_events = war_events_df
        except FileNotFoundError:
            print("Error: War events data file not found.")
            self.war_events = pd.DataFrame()
        except Exception as e:
            print(f"Unexpected error while loading war events data: {e}")
            self.war_events = pd.DataFrame()

        try:
            # Load Ukrainian borders shapefile
            self.ukraine_borders = gpd.read_file(get_path("border_data_dir"))
            self.ukraine_borders.set_crs(epsg=get_parameter('border_epsg'), inplace=True)
        except FileNotFoundError:
            print("Error: Ukrainian borders shapefile not found.")
            self.ukraine_borders = gpd.GeoDataFrame()
        except Exception as e:
            print(f"Unexpected error while loading Ukrainian borders shapefile: {e}")
            self.ukraine_borders = gpd.GeoDataFrame()

        try:
            # Load Russian-occupied territories shapefile
            self.rus_adv = gpd.read_file(get_path("rus_control_data_dir"))
            self.rus_adv.to_crs(epsg=get_parameter('border_epsg'), inplace=True)
        except FileNotFoundError:
            print("Error: Russian-occupied territories shapefile not found.")
            self.rus_adv = gpd.GeoDataFrame()
        except Exception as e:
            print(f"Unexpected error while loading Russian-occupied territories shapefile: {e}")
            self.rus_adv = gpd.GeoDataFrame()

    def plot_fire_data(self, data, event='', start_date=None, end_date=None, colors=['red'], alphas=[0.25], legend=False,
                       markersizes=[1], plot_cities=False, city_color='yellow', cmaps=None, column=None, plot_rus_adv=False):
        """
        Plot fire data within Ukrainian borders.

        Parameters
        ----------
        data : list of pd.DataFrame
            List of dataframes containing fire data with longitude and latitude columns.
        event : str, optional
            Text description for the event.
        start_date : dt.datetime, optional
            Start date for the event.
        end_date : dt.datetime, optional
            End date for the event.
        colors : list of str, optional
            List of colors for each dataset.
        alphas : list of float, optional
            List of alpha values (transparency) for each dataset.
        legend : bool, optional
            Whether to show a colorbar legend for the data points.
        markersizes : list of int, optional
            List of marker sizes for each dataset.
        plot_cities : bool, optional
            Whether to plot major cities on the map.
        city_color : str, optional
            Color for major cities.
        cmaps : list of str, optional
            List of colormap names for each dataset.
            If provided, the data points will be colored based on the values in the columns parameter.
        column : str, optional
            Column name to use for coloring the data points.
        plot_rus_adv : bool, optional
            Whether to plot Russian-occupied territories in Ukraine.

        Returns
        -------
        str
            The path to the saved plot.

        Raises
        ------
        Exception
            If an error occurs while plotting the data.
        """
        try:
            # Create plot and plot Ukrainian borders
            _, ax = plt.subplots(figsize=(15, 9))
            self.ukraine_borders.plot(ax=ax, color='white', edgecolor='black')
            # Plot Russian-occupied territories if specified
            if plot_rus_adv:
                self.rus_adv.boundary.plot(ax=ax, color='red', linewidth=1)
                self.rus_adv.plot(ax=ax, color='none', edgecolor='red', hatch='//')

            vmin, vmax = 0, 0
            for i, df in enumerate(data):
                if start_date and end_date:
                    df['ACQ_DATE'] = pd.to_datetime(df['ACQ_DATE']).dt.date
                    df = df[(df['ACQ_DATE'] >= start_date) & (df['ACQ_DATE'] <= end_date)]
                if column:
                    df = df.sort_values(by=column)
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']))
                gdf.set_crs(epsg=get_parameter('border_epsg'), inplace=True)
                if column:
                    if cmaps:
                        gdf.plot(ax=ax, column=column, cmap=cmaps, markersize=markersizes[i], alpha=alphas[i], legend=False)
                    else:
                        gdf.plot(ax=ax, column=column, color=colors[i], markersize=markersizes[i], alpha=alphas[i], legend=False)
                    vmin_i, vmax_i = gdf[column].min(), gdf[column].max()
                    vmin = min(vmin, vmin_i)
                    vmax = max(vmax, vmax_i)
                else:
                    gdf.plot(ax=ax, color=colors[i], markersize=markersizes[i], alpha=alphas[i], legend=False)

            # Plot major cities if specified
            if plot_cities:
                for city, (lat, lon, txt_lat, txt_lon) in self.major_cities.items():
                    plt.plot(lon, lat, marker='s', color=city_color, markersize=7)
                    plt.text(lon + txt_lon, lat + txt_lat, city, fontsize=9, color='white', weight='bold', 
                             path_effects=[pe.withStroke(linewidth=4, foreground="black")])

            # Set plot title based on event and date range
            if start_date and end_date:
                plt.title(f"{event}\n({start_date} bis {end_date})", fontsize=16, weight='bold')
            else:
                plt.title(event, fontsize=16, weight='bold')
            plt.axis('on')
            # Add grid for context
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            # Add grid titles
            ax.set_xlabel('Östliche Länge (°)')
            ax.set_ylabel('Nördliche Breite (°)')

            # Add a colorbar if a column is specified for coloring
            if column and legend and cmaps:
                sm = plt.cm.ScalarMappable(cmap=cmaps, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm._A = []
                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.set_ylabel(column)
                cbar.ax.set_position([0.85, 0.3, 0.03, 0.4])

            # Save the figure
            output_dir = get_path("output_plots_dir")
            dpi=500
            if start_date and end_date:
                if column:
                    file_path = os.path.join(output_dir, f"{start_date}_{end_date}_{column}.png")
                else:
                    file_path = os.path.join(output_dir, f"{start_date}_{end_date}.png")
            else:
                if column:
                    file_path = os.path.join(output_dir, f"{event}_{column}.png")
                else:
                    file_path = os.path.join(output_dir, f"{event}.png")
            plt.savefig(file_path, dpi=dpi)
            # plt.show()
            plt.close()
            return file_path
        except Exception as e:
            print(f"Error while plotting fire data: {e}")

    def plot_histogram(self, normal_scores, abnormal_scores, start_date, end_date, event, column):
        """
        Plot histograms for significance scores of normal and abnormal fires.

        Parameters
        ----------
        normal_scores : array-like
            Significance scores for normal fires.
        abnormal_scores : array-like
            Significance scores for abnormal fires.
        start_date : str or datetime.date
            Start date for the event. Expected format: 'YYYY-MM-DD'.
        end_date : str or datetime.date
            End date for the event. Expected format: 'YYYY-MM-DD'.
        event : str
            Text description for the event.
        column : str
            Column name to use for plotting the histogram.

        Returns
        -------
        str
            The path to the saved histogram plot.

        Raises
        ------
        Exception
            If an error occurs while plotting the histogram.
        """
        try:
            # Create histogram plots for normal and abnormal fire significance scores
            plt.figure(figsize=(12, 5))

            # Plot normal fires
            plt.subplot(1, 2, 1)
            bin_width = 0.05
            bins = int((normal_scores[column].max() - normal_scores[column].min()) / bin_width)
            plt.hist(normal_scores[column], bins=bins, color='blue', alpha=0.7)
            plt.title('Nicht kriegsbedingte Brände', fontsize=14)
            plt.xlabel('Signifikanz')
            plt.ylabel('Häufigkeit')
            plt.xlim(-1, 0)

            # Plot abnormal fires
            plt.subplot(1, 2, 2)
            bin_width = 0.05
            bins = int((normal_scores[column].max() - normal_scores[column].min()) / bin_width)
            plt.hist(abnormal_scores[column], bins=bins, color='red', alpha=0.7)
            plt.title('Kriegsbedingte Brände', fontsize=14)
            plt.xlabel('Signifikanz')
            plt.ylabel('Häufigkeit')
            plt.xlim(0, 1)

            # Set main title and save the figure
            plt.suptitle(f"{event}\n({start_date} bis {end_date})", fontsize=16, weight='bold')
            plt.tight_layout()
            output_dir = get_path("output_plots_dir")
            file_path = os.path.join(output_dir, f"{start_date}_{end_date}_{column}_histogram.png")
            plt.savefig(file_path, dpi=500)
            # plt.show()
            plt.close()
            # Return the path to the saved file
            return file_path
        except Exception as e:
            print(f"Error while plotting histogram: {e}")


if __name__ == "__main__":
    try:
        # Load example data for plotting
        predictions = pd.read_csv('results/50km/test_predictions.csv')
        abnormal_fires = predictions[predictions['ABNORMAL_LABEL_DECAY'] == True]
        normal_fires = predictions[predictions['ABNORMAL_LABEL_DECAY'] == False]

        # Initialize the UkraineFirePlotting object
        plotting = UkraineFirePlotting()

        # Plotting example using war events
        for _, row in plotting.war_events.iterrows():
            start_date, end_date, event = pd.to_datetime(row['START_DATE']).date(), pd.to_datetime(row['END_DATE']).date(), row['EVENT']
            # Plot abnormal and normal fires
            plotting.plot_fire_data([normal_fires, abnormal_fires], event=event, start_date=start_date, end_date=end_date,
                                               colors=['black', 'red'], alphas=[0.05, 0.1], markersizes=[2, 4],
                                               plot_cities=True, city_color='blue', plot_rus_adv=True)
            plotting.plot_fire_data([abnormal_fires], event=event, start_date=start_date, end_date=end_date,
                                               cmaps='Reds', alphas=[0.2], markersizes=[7], column='SIGNIFICANCE_SCORE_DECAY',
                                               plot_cities=True, city_color='blue', legend=True, plot_rus_adv=True)
            # Plot significance score histogram
            plotting.plot_histogram(normal_fires, abnormal_fires, start_date, end_date, event, column='SIGNIFICANCE_SCORE')
            plotting.plot_histogram(normal_fires, abnormal_fires, start_date, end_date, event, column='SIGNIFICANCE_SCORE_DECAY')

            print(f"Plotted data for event: {event} ({start_date} - {end_date})")

        # Plot population density
        try:
            population_data = pd.read_csv('data/ukr_pop_density/ukr_pop_density_2020_1km.csv')
            # Assign all rows with a value greater than 250 to 250
            population_data['POP_DENSITY'] = population_data['POP_DENSITY'].apply(lambda x: 250 if x > 250 else x)
            plotting.plot_fire_data([population_data], event='Bevölkerungsdichte in der Ukraine', cmaps='viridis', alphas=[0.25], 
                                    markersizes=[1], column='POP_DENSITY', legend=True, plot_cities=True, city_color='yellow')     
            print("Plotted population density in Ukraine")
        except FileNotFoundError:
            print("Error: Population density data file not found.")

        # Plot land use in Ukraine 1km resolution
        try:
            land_use_data = pd.read_csv('data/ukr_land_use/ukr_land_use_2022_1km.csv')
            land_use_class_mapping = {
                'LAND_USE_CLASS_0': 0,
                'LAND_USE_CLASS_1': 1,
                'LAND_USE_CLASS_2': 2,
                'LAND_USE_CLASS_3': 3,
                'LAND_USE_CLASS_4': 4
            }
            # Create a custom color map
            custom_cmap = ListedColormap(['darkblue', 'green', 'brown', 'orange', 'red'])

            # Apply the custom color map to the plotting function
            land_use_data['LAND_USE_CLASS'] = land_use_data[['LAND_USE_CLASS_0', 'LAND_USE_CLASS_1', 'LAND_USE_CLASS_2', 
                                                                'LAND_USE_CLASS_3', 'LAND_USE_CLASS_4']].idxmax(axis=1).map(land_use_class_mapping)
            plotting.plot_fire_data([land_use_data], event='Landnutzung in der Ukraine', cmaps=custom_cmap, alphas=[0.25], 
                                    markersizes=[1], column='LAND_USE_CLASS', legend=True, plot_cities=True, city_color='red')
            print("Plotted land use in Ukraine 1km resolution")
        except FileNotFoundError:
            print("Error: Land use data file (1km) not found.")

        # Plot land use in Ukraine 10km resolution
        try:
            land_use_data = pd.read_csv('data/ukr_land_use/ukr_land_use_2022_10km.csv')
            land_use_data['LAND_USE_CLASS'] = land_use_data[['LAND_USE_CLASS_0', 'LAND_USE_CLASS_1', 'LAND_USE_CLASS_2', 
                                                             'LAND_USE_CLASS_3', 'LAND_USE_CLASS_4']].idxmax(axis=1).map(land_use_class_mapping)
            plotting.plot_fire_data([land_use_data], event='10km Gitternetz', colors=['black'], alphas=[1], 
                                    markersizes=[5], column='LAND_USE_CLASS', legend=True, plot_cities=True, city_color='red')
            print("Plotted land use in Ukraine 10km resolution")
        except FileNotFoundError:
            print("Error: Land use data file (10km) not found.")

        # Plot land use in Ukraine 50km resolution
        try:
            land_use_data = pd.read_csv('data/ukr_land_use/ukr_land_use_2022_50km.csv')
            land_use_data['LAND_USE_CLASS'] = land_use_data[['LAND_USE_CLASS_0', 'LAND_USE_CLASS_1', 'LAND_USE_CLASS_2', 
                                                             'LAND_USE_CLASS_3', 'LAND_USE_CLASS_4']].idxmax(axis=1).map(land_use_class_mapping)
            plotting.plot_fire_data([land_use_data], event='50km Gitternetz', colors=['black'], alphas=[1], 
                                    markersizes=[5], column='LAND_USE_CLASS', legend=True, plot_cities=True, city_color='red')
            print("Plotted land use in Ukraine 50km resolution")
        except FileNotFoundError:
            print("Error: Land use data file (50km) not found.")
    except FileNotFoundError as e:
        print(f"Error in main: {e}")
    except Exception as e:
        print(f"Unexpected error in main: {e}")