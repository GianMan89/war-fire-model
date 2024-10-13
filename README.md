# Forest Fire Classification in Ukraine

This repository provides an implementation for classifying forest fires in Ukraine between February 2022 and September 2024 as potentially war-related or natural fires. This work extends *The Economist's* Ukraine war-fire model, which uses machine learning techniques and publicly available satellite data to detect war events.

## Project Overview

The primary objective of this project is to enhance the classification of forest fires in Ukraine during the ongoing conflict. Leveraging machine learning techniques, this project aims to distinguish between natural and war-related fire events. The original work by *The Economist* detected thousands of such events, and this repository builds on those foundations by incorporating additional data, improved models, and new approaches for explainability.

Key features of the project include:
- Training on historical natural fire data to build a model for classifying recent fire events.
- Utilizing explainable AI (XAI) methods to provide transparency in model predictions.
- Generating visualizations for insights into fire patterns and relationships with population density and land use.
- Extending *The Economist's* work by adding active learning, clustering, and graph-based methods for enhanced detection and prediction accuracy.

## Repository Structure

The repository is organized as follows:

- **config/**: Configuration files for the project.
  - **config.py**: Configuration settings for the pipeline.
  - **ukr_weather_api_call_data.csv**: Coordinates for weather API calls.
- **data/**: Contains various datasets related to the project.
  - **ukr_borders/**: Data related to Ukraine's borders.
  - **ukr_fires/**: Fire event data.
  - **ukr_land_use/**: Land use data for Ukraine.
  - **ukr_oblasts/**: Data on Ukrainian oblasts.
  - **ukr_pop_density/**: Population density data.
  - **ukr_war_events/**: Data related to war events.
  - **ukr_weather/**: Weather data for Ukraine.
- **models/**: Directory to store trained models.
- **results/**: Stores model predictions and visualizations.
  - **50km/**: Contains CSV files with results at 50km resolution.
  - **plots/**: Subfolder with generated plots.
- **src/**: Source code for the project.
  - **plotting/**: Contains plotting utilities.
    - **plotting.py**: Plotting functions for visualization.
  - **preprocessing/**: Scripts for preprocessing data.
    - **feature_engineering.py**: Feature engineering scripts.
    - **load_data.py**: Data loading functions.
    - **update_data.py**: Functions to update data.
  - **models/**: Contains model-related scripts.
    - **error_threshold.py**: Handles error threshold calculations.
    - **explainer.py**: Functions for explainability.
    - **nearest_neighbor.py**: Nearest neighbor analysis.
    - **score_decay.py**: Handles score decay calculations.
  - **pipelines/**: Pipeline scripts for data and model processing.
    - **data_pipeline.py**: Data processing pipeline.
    - **model_pipeline.py**: Model training and inference pipeline.
- **notebook.ipynb**: Jupyter notebook showcasing the end-to-end pipeline, including training, testing, and visualization.
- **LICENSE**: License information for the project.
- **poetry.lock**: Lockfile for managing project dependencies.
- **pyproject.toml**: Project configuration and dependencies.
- **README.md**: Documentation for the project, including usage instructions and context.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Poetry for dependency management

To install the required libraries, run:
```sh
poetry install
```

### Running the Pipeline

1. **Data Loading and Preprocessing**: The data is loaded using the `DataPipeline` class, which preprocesses the data for model training and evaluation.

2. **Model Training**: The `FirePredictionPipeline` class is used to train the model on historical fire data.

3. **Inference and Evaluation**: After training, the model is saved and can be reloaded for inference on recent fire events. Predictions are saved for further analysis.

4. **Nearest Neighbor Analysis and Explainability**: The pipeline also includes functions for finding nearest neighbors and explaining individual predictions.

5. **Visualization**: The notebook includes plotting utilities to visualize the results, including fire locations, population density, and land use maps.

### Example Usage

Refer to the Jupyter notebook (`notebook.ipynb`) for an example of the complete pipeline, including data loading, model training, prediction, and visualization.

## Visualization

The project includes a range of visualizations to help understand the model's performance and the relationship between fires, population density, and land use.

- **Fire Data Plots**: Visualize normal and abnormal fires over specific events and timeframes.
- **Population Density and Land Use Maps**: Visualize population density and land use across Ukraine to understand how these factors correlate with fire events.

## Contributions

Contributions are welcome! If you'd like to improve the pipeline, add new features, or suggest changes, feel free to create an issue or submit a pull request.

## Relation to *The Economist's* Ukraine War-Fire Model

This project builds upon *The Economist's* Ukraine war-fire model, which used statistical techniques and satellite temperature anomaly data to detect war-related fire events. You can find the original repository and data here: [Tracking the Ukraine war: where is the latest fighting?](https://www.economist.com/interactive/graphic-detail/ukraine-fires).

The original work detected over 93,000 fire events between February 24th, 2022, and June 19th, 2024. This repository extends that work by:
- Adding additional machine learning methods for classification.
- Including temporal and clustering analysis for enhanced detection.
- Incorporating XAI techniques for better understanding of model predictions.

## Questions or Missing Information

Please let me know if the following information needs to be added or clarified:
- **Data Sources**: Details about where the fire, population density, and land use data were obtained.
- **Model Information**: More specifics on the type of model(s) used in the `FirePredictionPipeline`.
- **Hyperparameters**: If there are specific hyperparameters used for training that should be documented.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, please reach out at [your-email@example.com].