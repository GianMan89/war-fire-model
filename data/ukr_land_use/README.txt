Download links for raw data files:
  "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc2022/34U_20220101-20230101.tif",
  "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc2022/35U_20220101-20230101.tif",
  "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc2022/36U_20220101-20230101.tif",
  "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc2022/37U_20220101-20230101.tif",
  "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc2022/35T_20220101-20230101.tif",
  "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc2022/36T_20220101-20230101.tif",
  "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc2022/37T_20220101-20230101.tif"

Data has been preprocessed to reduce resolution from 10m to 200m.

Original class labels: 0: 'No data', 1: 'Water', 2: 'Trees', 4: 'Flooded vegetation', 5: 'Crops', 7: 'Built Area', 8: 'Bare ground', 9: 'Snow and ice', 10: 'Clouds', 11: 'Rangeland'
Original class labels have been aggregated to the following new class labels:
0: ['No data', 'Water', 'Snow and ice', 'Clouds'], 
1: ['Trees'], 
2: ['Flooded vegetation', 'Bare ground', 'Rangeland'], 
3: ['Crops'], 
4: ['Built Area']

========================================================
README: Sentinel-2 10m land use/land cover time series of the world. Produced by Impact Observatory and Esri.
========================================================

Description
This layer displays a global map of land use/land cover (LULC) derived from ESA Sentinel-2 imagery at 10m resolution. Each year is generated with Impact Observatory’s deep learning AI land classification model, trained using billions of human-labeled image pixels from the National Geographic Society. The global maps are produced by applying this model to the Sentinel-2 Level-2A image collection on Microsoft’s Planetary Computer, processing over 400,000 Earth observations per year.

The algorithm generates LULC predictions for nine classes, described in detail below.  

The year 2017 has a land cover class assigned for every pixel, but its class is based upon fewer images than the other years. The years 2018-2023 are based upon a more complete set of imagery. For this reason, the year 2017 may have less accurate land cover class assignments than the years 2018-2023.

Variable mapped: Land use/land cover in 2017, 2018, 2019, 2020, 2021, 2022, 2023
Source Data Coordinate System: Universal Transverse Mercator (UTM) WGS84
Service Coordinate System: Web Mercator Auxiliary Sphere WGS84 (EPSG:3857)
Extent: Global
Source imagery: Sentinel-2 L2A
Cell Size: 10-meters
Type: Thematic
Attribution: Esri, Impact Observatory

What can you do with this layer?

Global land use/land cover maps provide information on conservation planning, food security, and hydrologic modeling, among other things. This dataset can be used to visualize land use/land cover anywhere on Earth.
 
This layer can also be used in analyses that require land use/land cover input. For example, the Zonal toolset allows a user to understand the composition of a specified area by reporting the total estimates for each of the classes. 

NOTE: Land use focus does not provide the spatial detail of a land cover map. As such, for the built area classification, yards, parks, and groves will appear as built area rather than trees or rangeland classes.

Class definitions

Value	Name	Description
1	Water	Areas where water was predominantly present throughout the year; may not cover areas with sporadic or ephemeral water; contains little to no sparse vegetation, no rock outcrop nor built up features like docks; examples: rivers, ponds, lakes, oceans, flooded salt plains.
2	Trees	Any significant clustering of tall (~15 feet or higher) dense vegetation, typically with a closed or dense canopy; examples: wooded vegetation,  clusters of dense tall vegetation within savannas, plantations, swamp or mangroves (dense/tall vegetation with ephemeral water or canopy too thick to detect water underneath).
4	Flooded vegetation	Areas of any type of vegetation with obvious intermixing of water throughout a majority of the year; seasonally flooded area that is a mix of grass/shrub/trees/bare ground; examples: flooded mangroves, emergent vegetation, rice paddies and other heavily irrigated and inundated agriculture.
5	Crops	Human planted/plotted cereals, grasses, and crops not at tree height; examples: corn, wheat, soy, fallow plots of structured land.
7	Built Area	Human made structures; major road and rail networks; large homogenous impervious surfaces including parking structures, office buildings and residential housing; examples: houses, dense villages / towns / cities, paved roads, asphalt.
8	Bare ground	Areas of rock or soil with very sparse to no vegetation for the entire year; large areas of sand and deserts with no to little vegetation; examples: exposed rock or soil, desert and sand dunes, dry salt flats/pans, dried lake beds, mines.
9	Snow/Ice	Large homogenous areas of permanent snow or ice, typically only in mountain areas or highest latitudes; examples: glaciers, permanent snowpack, snow fields.
10	Clouds	No land cover information due to persistent cloud cover.
11	Rangeland	Open areas covered in homogenous grasses with little to no taller vegetation; wild cereals and grasses with no obvious human plotting (i.e., not a plotted field); examples: natural meadows and fields with sparse to no tree cover, open savanna with few to no trees, parks/golf courses/lawns, pastures. Mix of small clusters of plants or single plants dispersed on a landscape that shows exposed soil or rock; scrub-filled clearings within dense forests that are clearly not taller than trees; examples: moderate to sparse cover of bushes, shrubs and tufts of grass, savannas with very sparse grasses, trees or other plants.

Classification Process

These maps include Version 003 of the global Sentinel-2 land use/land cover data product. It is produced by a deep learning model trained using over five billion hand-labeled Sentinel-2 pixels, sampled from over 20,000 sites distributed across all major biomes of the world.

The underlying deep learning model uses 6-bands of Sentinel-2 L2A surface reflectance data: visible blue, green, red, near infrared, and two shortwave infrared bands. To create the final map, the model is run on multiple dates of imagery throughout the year, and the outputs are composited into a final representative map for each year.

The input Sentinel-2 L2A data was accessed via Microsoft’s Planetary Computer and scaled using Microsoft Azure Batch.


Citation
Karra, Kontgis, et al. “Global land use/land cover with Sentinel-2 and deep learning.” IGARSS 2021-2021 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2021.

Acknowledgements

Training data for this project makes use of the National Geographic Society Dynamic World training dataset, produced for the Dynamic World Project by National Geographic Society in partnership with Google and the World Resources Institute.