import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore")

# Add the project root to the Python path
try:
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
    sys.path.append(project_root)
except Exception as e:
    raise RuntimeError(f"Failed to add project root to Python path: {e}")

class OneNearestNeighborModel:
    """
    OneNearestNeighborModel
    A model that uses the nearest neighbor algorithm to find the closest samples for fire days and non-fire days.

    Attributes
    ----------
    scaler : StandardScaler
        Scaler used to standardize the features.
    nearest_neighbors_nofiredays : sklearn.neighbors.NearestNeighbors
        NearestNeighbors model for non-fire days.
    nearest_neighbors_firedays : sklearn.neighbors.NearestNeighbors
        NearestNeighbors model for fire days.
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
        self.nearest_neighbors_nofiredays = None
        self.nearest_neighbors_firedays = None
        self.X_firedays = None
        self.X_nofiredays = None
        self.y_firedays = None
        self.y_nofiredays = None

    def fit(self, X, y):
        """
        Fit the model using the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values (fire days and non-fire days).
        
        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        Exception
            If an error occurs during fitting the model.
        """
        try:
            # Separate the input data into fire days and non-fire days
            self.X_firedays = X[y > 0]
            self.X_nofiredays = X[y == 0]
            self.y_firedays = y[y > 0]
            self.y_nofiredays = y[y == 0]

            # Standardize the features using StandardScaler
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
            X_scaled_firedays = X_scaled[y > 0]
            X_scaled_nofiredays = X_scaled[y == 0]

            # Fit nearest neighbors model
            self.nearest_neighbors_firedays = self.fit_nn(X_scaled_firedays) 
            self.nearest_neighbors_nofiredays = self.fit_nn(X_scaled_nofiredays)
        
        except Exception as e:
            print(f"Error during fitting the model: {e}")
        
        return self

    def transform(self, X):
        """
        Transform the input data by scaling and finding nearest neighbors.

        Parameters
        ----------
        X : pandas.DataFrame
            The input features to be transformed.
        
        Returns
        -------
        X_nn_firedays : pandas.DataFrame
            DataFrame containing the nearest neighbors for all fire days for each sample in X.
        X_nn_nofiredays : pandas.DataFrame
            DataFrame containing the nearest neighbors for all days for each sample in X.
        y_nn_firedays : pandas.DataFrame
            DataFrame containing the target values for the nearest neighbors for all fire days.
        y_nn_nofiredays : pandas.DataFrame
            DataFrame containing the target values for the nearest neighbors for all days.

        Raises
        ------
        Exception
            If an error occurs during transforming the data.
        """
        try:
            # Scale the input features using the fitted scaler
            X_scaled = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)
            X_nn_firedays = np.zeros(X_scaled.shape)
            X_nn_nofiredays = np.zeros(X_scaled.shape)
            y_nn_firedays = np.zeros(X_scaled.shape[0])
            y_nn_nofiredays = np.zeros(X_scaled.shape[0])

            # Get the nearest neighbors
            _, nn_firedays = self.nearest_neighbors_firedays.kneighbors(X_scaled)
            X_nn_firedays = self.X_firedays.iloc[nn_firedays.flatten()]
            y_nn_firedays = self.y_firedays.iloc[nn_firedays.flatten()]

            _, nn_nofiredays = self.nearest_neighbors_nofiredays.kneighbors(X_scaled)
            X_nn_nofiredays = self.X_nofiredays.iloc[nn_nofiredays.flatten()]
            y_nn_nofiredays = self.y_nofiredays.iloc[nn_nofiredays.flatten()]

        except Exception as e:
            print(f"Error during transforming the data: {e}")

        # Reset the index of the DataFrames and rename the columns
        X_nn_firedays = X_nn_firedays.reset_index(drop=True)
        X_nn_nofiredays = X_nn_nofiredays.reset_index(drop=True)
        y_nn_firedays = y_nn_firedays.reset_index(drop=True)
        y_nn_nofiredays = y_nn_nofiredays.reset_index(drop=True)
        y_nn_firedays.columns = ['FIRE_COUNT_CELL']
        y_nn_nofiredays.columns = ['FIRE_COUNT_CELL']

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

        Raises
        ------
        Exception
            If an error occurs during fitting the NearestNeighbors
        """
        try:
            # Initialize and fit the nearest neighbors model
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(X)
            return nn
        except Exception as e:
            print(f"Error during fitting NearestNeighbors model: {e}")
            return None


if __name__ == "__main__":
    # Example data
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    }
    target = [0, 0, 0, 1, 1, 0, 0, 1, 1, 0]

    # Convert to DataFrame
    X = pd.DataFrame(data)
    y = pd.Series(target)

    # Initialize and fit the model
    model = OneNearestNeighborModel()
    model.fit(X, y)

    # Transform the data
    X_nn_firedays_df, X_nn_nofiredays_df, y_nn_firedays_df, y_nn_nofiredays_df = model.transform(X)

    # Print the results
    print("Nearest neighbors for fire days (features):")
    print(X_nn_firedays_df)
    print("\nNearest neighbors for non-fire days (features):")
    print(X_nn_nofiredays_df)
    print("\nNearest neighbors for fire days (target):")
    print(y_nn_firedays_df)
    print("\nNearest neighbors for non-fire days (target):")
    print(y_nn_nofiredays_df)