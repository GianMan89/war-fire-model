import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

class OneNearestNeighborModel:
    """
    OneNearestNeighborModel
    A model that uses the nearest neighbor algorithm to find the closest samples for fire days and non-fire days.

    Attributes
    ----------
    scaler : StandardScaler
        Scaler used to standardize the features.
    nearest_neighbors_nofiredays : dict
        Dictionary storing nearest neighbors models for all days, keyed by grid cell.
    nearest_neighbors_firedays : dict
        Dictionary storing nearest neighbors models for fire days, keyed by grid cell.
    ids_nofiredays : DataFrame
        DataFrame containing the 'GRID_CELL' information for all days.
    ids_firedays : DataFrame
        DataFrame containing the 'GRID_CELL' information for fire days.
    X_firedays : array-like of shape (n_samples, n_features)
        Scaled input samples for fire days.
    X_nofiredays : array-like of shape (n_samples, n_features)
        Scaled input samples for all days.
    y_firedays : array-like of shape (n_samples,)
        Target values for fire days.
    y_nofiredays : array-like of shape (n_samples,)
        Target values for all days.
        
    Methods
    ----------
    fit(X, y, ids)
    transform(X, ids)
    fit_nn(X)
    """

    def __init__(self):
        self.scaler = None
        self.nearest_neighbors_nofiredays = {}
        self.nearest_neighbors_firedays = {}
        self.ids_nofiredays = None
        self.ids_firedays = None
        self.X_firedays = None
        self.X_nofiredays = None
        self.y_firedays = None
        self.y_nofiredays = None

    def fit(self, X, y, ids):
        """
        Fit the model using the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values (fire days and non-fire days).
        ids : DataFrame
            DataFrame containing the 'GRID_CELL' column which identifies the grid cell for each sample.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.X_firedays = X[y > 0]
        self.X_nofiredays = X[y == 0]
        self.y_firedays = y[y > 0]
        self.y_nofiredays = y[y == 0]
        self.ids_firedays = ids[y > 0]
        self.ids_nofiredays = ids[y == 0]

        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        X_scaled_firedays = X_scaled[y > 0]
        X_scaled_nofiredays = X_scaled[y == 0]
        
        for grid_cell in ids['GRID_CELL'].unique():
            indices_firedays = self.ids_firedays['GRID_CELL'][self.ids_firedays['GRID_CELL'] == grid_cell].index
            indices_nofiredays = self.ids_nofiredays['GRID_CELL'][self.ids_nofiredays['GRID_CELL'] == grid_cell].index
            if len(indices_firedays) == 0:
                self.nearest_neighbors_firedays[grid_cell] = None
            if len(indices_nofiredays) == 0:
                self.nearest_neighbors_nofiredays[grid_cell] = None
            if len(indices_firedays) > 0:
                self.nearest_neighbors_firedays[grid_cell] = self.fit_nn(X_scaled_firedays.loc[indices_firedays])
            if len(indices_nofiredays) > 0:
                self.nearest_neighbors_nofiredays[grid_cell] = self.fit_nn(X_scaled_nofiredays.loc[indices_nofiredays]) 
        return self

    def transform(self, X, ids):
        """
        Transform the input data by scaling and finding nearest neighbors.

        Parameters
        ----------
        X : pandas.DataFrame
            The input features to be transformed.
        ids : pandas.DataFrame
            DataFrame containing the 'GRID_CELL' information for each sample in X.
        Returns
        -------
        X_nn_firedays : list of pandas.DataFrame
            List of DataFrames containing the nearest neighbors for fire days for each sample in X.
        X_nn_nofiredays : list of pandas.DataFrame
            List of DataFrames containing the nearest neighbors for all days for each sample in X.
        y_nn_firedays : list of pandas.DataFrame
            List of DataFrames containing the target values for the nearest neighbors for fire days.
        y_nn_nofiredays : list of pandas.DataFrame
            List of DataFrames containing the target values for the nearest neighbors for all days.
        """
        X_scaled = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)
        X_nn_firedays = np.zeros(X_scaled.shape)
        X_nn_nofiredays = np.zeros(X_scaled.shape)
        y_nn_firedays = np.zeros(X_scaled.shape[0])
        y_nn_nofiredays = np.zeros(X_scaled.shape[0])

        # Iterate over unique grid cells
        unique_grid_cells = ids['GRID_CELL'].unique()
        for grid_cell in unique_grid_cells:
            # Select relevant test samples for the current grid cell
            indices = ids[ids['GRID_CELL'] == grid_cell].index
            X_group = X_scaled.loc[indices]
            # Get the nearest neighbors for the current group of samples
            nn_firedays_model = self.nearest_neighbors_firedays.get(grid_cell)
            nn_nofiredays_model = self.nearest_neighbors_nofiredays.get(grid_cell)
            _, nn_firedays = nn_firedays_model.kneighbors(X_group.values)
            _, nn_nofiredays = nn_nofiredays_model.kneighbors(X_group.values)
            # Save the results at the appropriate indices
            X_nn_firedays[indices] = self.X_firedays.iloc[nn_firedays.flatten()]
            X_nn_nofiredays[indices] = self.X_nofiredays.iloc[nn_nofiredays.flatten()]
            y_nn_firedays[indices] = self.y_firedays.iloc[nn_firedays.flatten()]
            y_nn_nofiredays[indices] = self.y_nofiredays.iloc[nn_nofiredays.flatten()]
        return X_nn_firedays, X_nn_nofiredays, y_nn_firedays, y_nn_nofiredays
    
    @staticmethod
    def fit_nn(X):
        """
        Fit a NearestNeighbors model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        Returns
        -------
        nn : NearestNeighbors
            The fitted NearestNeighbors model.
        """
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X)
        return nn