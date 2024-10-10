import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from quantile_forest import RandomForestQuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

from src.preprocessing.load_data import DataLoader
from src.training.feature_engineering import FeatureEngineering
from config.config import get_parameter
from utils.file_utils import get_path, save_large_model, load_large_model

class ThresholdStep(BaseEstimator, TransformerMixin):
    """
    ThresholdStep is a custom transformer that determines the best threshold for classification based on the provided predictions and true values.
    
    Attributes
    ----------
    threshold : float, optional
        Initial threshold value. If not provided, it will be determined during the fitting process.
    Attributes
    threshold : float
        The best threshold value determined during the fitting process.
    parts_number : None
        Placeholder attribute, not used in the current implementation.
    
    Methods
    ----------
    fit(y_pred, y_true=None, y_labels=None)
        Fits the model by determining the best threshold for classification based on the provided predictions and true values.
    transform(y_pred, y_true=None)
        Transform the predicted values by comparing them with the true labels.
    """
    def __init__(self, threshold=None):
        self.threshold = threshold

    def fit(self, y_pred, y_true=None, y_labels=None):
        """
        Fits the model by determining the best threshold for classification based on the provided predictions and true values.

        Parameters:
        -----------
        y_pred : array-like
            Predicted values from the model.
        y_true : array-like, optional
            True values for the target variable. Must be provided unless `y_labels` is a float.
        y_labels : array-like or float, optional
            Labels for the target variable. If a float is provided, it represents the fraction of total fires to be used as a threshold.
        Raises:
        -------
        ValueError
            If `y_true` or `y_labels` is not provided.
        Returns:
        --------
        self : object
            Returns the instance itself.
        """

        if y_true is None or y_labels is None:
            raise ValueError('Transformer requires y_true and y_labels to be passed.')
        
        error = y_true - y_pred
        error[error < 0] = 0

        if type(y_labels) == float:
            sorted_indices = np.argsort(error).reset_index(drop=True)
            sorted_y_val = y_true.iloc[sorted_indices]
            cumulative_fires = np.cumsum(sorted_y_val)
            total_fires = np.sum(y_true)
            threshold_index = np.searchsorted(cumulative_fires, y_labels * total_fires)
            threshold = error.iloc[sorted_indices[threshold_index]]
            y_labels = (error > threshold).astype(int)
            self.parts_number = None
        
        best_threshold = None
        highest_balanced_accuracy = 0
        thresholds = np.linspace(error.min(), error.max(), 1000)

        for threshold in thresholds:
            y_pred_threshold = (error > threshold).astype(int)
            balanced_acc = balanced_accuracy_score(y_labels, y_pred_threshold)
            if balanced_acc > highest_balanced_accuracy:
                highest_balanced_accuracy = balanced_acc
                best_threshold = threshold

        self.threshold = best_threshold
        return self

    def transform(self, y_pred, y_true=None):
        """
        Transform the predicted values by comparing them with the true labels.
        This method calculates the error between the predicted and true values,
        determines whether the error is abnormal based on a predefined threshold,
        and computes a significance score for each prediction.

        Parameters
        ----------
        y_pred : array-like of shape (n_samples,)
            The predicted values.
        y_true : array-like of shape (n_samples,), optional
            The true labels. If not provided, a ValueError is raised.
        Returns
        -------
        transformed : ndarray of shape (2, n_samples)
            A 2D array where the first row contains binary values indicating
            whether the error is abnormal (1) or not (0), and the second row
            contains the significance scores for each prediction.
        Raises
        ------
        ValueError
            If the threshold is not set or if the true labels are not provided.
        """

        if self.threshold is None:
            raise ValueError("Threshold is not set. Please fit the transformer first.")
        if y_true is None:
            raise ValueError("True labels are required for the transformation.")
        
        error = y_true - y_pred
        error[error < 0] = 0
        is_abnormal = error > self.threshold
        significance_score = np.array(pd.Series(error).apply(lambda value: ((value - self.threshold) / self.threshold) if value < self.threshold else (value - self.threshold) / value))
        return np.array([is_abnormal.astype(int), significance_score])

class RecalculateConfidenceScores(BaseEstimator, TransformerMixin):
    """
    A custom transformer that recalculates confidence scores based on a sigmoid decay function.
    This transformer adjusts confidence scores of events based on their temporal distance from the last significant event in the same grid cell.

    Attributes
    ----------
    decay_rate : float
        The rate at which the confidence score decays over time.
    midpoint : float
        The midpoint of the sigmoid function.
    cutoff : float
        The time difference beyond which the confidence score is set to zero.

    Methods
    -------
    sigmoid_decay(time_diff)
        Applies a sigmoid decay function to the time difference.
    fit(X, y=None)
        Fits the transformer to the data. This method does nothing and is present for compatibility.
    transform(X, dates=None, grid_cells=None)
        Transforms the input data by recalculating the confidence scores based on the sigmoid decay function.
    """

    def __init__(self, decay_rate, midpoint, cutoff):
        self.decay_rate = decay_rate
        self.midpoint = midpoint
        self.cutoff = cutoff

    def sigmoid_decay(self, time_diff):
        """
        Apply a sigmoid decay function to a given time difference.

        Parameters
        ----------
        time_diff : float
            The time difference to which the sigmoid decay function is applied.

        Returns
        -------
        float
            The result of the sigmoid decay function. Returns 0 if `time_diff` is greater than `self.cutoff`, 
            otherwise returns a value between 0 and 1 based on the sigmoid function.
        """

        if time_diff > self.cutoff:
            return 0
        return 1 / (1 + np.exp(self.decay_rate * (time_diff - self.midpoint)))

    def fit(self, X, y=None):
        """
        Dummy method for compatibility with the scikit-learn pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values (ignored in this implementation).
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X, dates=None, grid_cells=None):
        """
        Recalculate confidence scores based on historical events and apply a decay function.

        Parameters
        ----------
        X : list of numpy arrays
            A list containing two numpy arrays:
            - y_pred: Predicted labels.
            - y_scores: Confidence scores associated with the predictions.
        dates : numpy array, optional
            Array of dates corresponding to each prediction. Required for recalculating confidence scores.
        grid_cells : numpy array, optional
            Array of grid cell identifiers corresponding to each prediction. Required for recalculating confidence scores.
        Returns
        -------
        list
            A list containing:
            - labels: Binary labels recalculated based on the confidence scores.
            - recalculated_confidences: Confidence scores after applying the decay function.
            - y_pred: Original predicted labels.
            - y_scores: Original confidence scores.
        Raises
        ------
        ValueError
            If `dates` or `grid_cells` is None.
        """

        if dates is None or grid_cells is None:
            raise ValueError("Dates and grid cells are required for the recalculation of confidence scores.")
        y_scores = X[1]
        y_pred = X[0]
        indexed_events = list(enumerate(zip(y_scores, dates, grid_cells)))
        indexed_events_sorted = sorted(indexed_events, key=lambda x: x[1][1])
        recalculated_confidences = [None] * len(y_scores)
        last_war_events = {}
        for i, (original_index, (current_conf, current_date, grid_cell)) in enumerate(indexed_events_sorted):
            if current_conf > 0:
                last_war_events[grid_cell] = {
                    'ACQ_DATE': current_date,
                    'SIGNIFICANCE_SCORE': current_conf
                }
                recalculated_confidences[original_index] = current_conf
            elif grid_cell in last_war_events:
                last_war_event = last_war_events[grid_cell]
                current_date = pd.to_datetime(current_date)
                last_war_event['ACQ_DATE'] = pd.to_datetime(last_war_event['ACQ_DATE'])
                time_diff = (current_date - last_war_event['ACQ_DATE']) / np.timedelta64(1, 'D')
                decayed_influence = self.sigmoid_decay(time_diff) * last_war_event['SIGNIFICANCE_SCORE']
                if decayed_influence > current_conf and decayed_influence > 0:
                    new_conf = decayed_influence
                else:
                    new_conf = current_conf
                recalculated_confidences[original_index] = new_conf
            else:
                recalculated_confidences[original_index] = current_conf
        
        recalculated_confidences = np.array(recalculated_confidences)
        labels = np.where(recalculated_confidences > 0, 1, 0)
        return [labels, recalculated_confidences, y_pred, y_scores]

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

class FirePredictionPipeline:
    """
    A pipeline for fire prediction using a combination of scaling, PCA, and a RandomForestQuantileRegressor.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The machine learning pipeline consisting of a scaler, PCA, and a quantile regressor.
    onn_model : OneNearestNeighborModel
        The One Nearest Neighbor model used for prediction.
    rf_fitted : bool
        Indicates whether the random forest model has been fitted.
    threshold_fitted : bool
        Indicates whether the threshold has been fitted.
    onn_fitted : bool
        Indicates whether the One Nearest Neighbor model has been fitted.
    Methods
    -------
    fit_rf(X_train, y_train)
        Fits the random forest model using the training data.
    fit_threshold(X_calib, y_calib)
        Fits the threshold model using the calibration data. Requires the random forest model to be fitted first.
    fit_onn(X_train, y_train, ids_train)
        Fits the One Nearest Neighbor model using the training data.
    predict(X_test, y_test, ids_calib)
        Predicts the fire occurrence and confidence scores using the test data and calibration IDs.
    save()
        Saves the pipeline to disk in parts.
    load()
        Loads the pipeline from disk. Requires the pipeline to be saved first.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=get_parameter("n_components"), random_state=get_parameter("random_state"))),
            ('regressor', RandomForestQuantileRegressor(n_estimators=get_parameter("n_estimators"), random_state=get_parameter("random_state"))),
        ])
        self.onn_model = OneNearestNeighborModel()
        self.rf_fitted = False
        self.threshold_fitted = False
        self.onn_fitted = False

    def fit_rf(self, X_train, y_train):
        """
        Fit the random forest model using the provided training data.

        Parameters
        ----------
        X_train : array-like or pandas DataFrame of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).
        Returns
        -------
        None
        """

        self.pipeline.fit(X_train, y_train)
        self.rf_fitted = True

    def fit_threshold(self, X_calib, y_calib):
        """
        Fit the threshold and recalculate confidence scores using the calibration data.
        This method appends a threshold step and a decay step to the pipeline, then fits the threshold 
        using the predicted and true calibration labels.

        Parameters
        ----------
        X_calib : array-like of shape (n_samples, n_features)
            The input samples for calibration.
        y_calib : array-like of shape (n_samples,)
            The true labels for calibration.
        Raises
        ------
        ValueError
            If the random forest model is not fitted before calling this method.
        Notes
        -----
        This method modifies the pipeline by adding a 'threshold' step and a 'decay' step.
        It also sets the `threshold_fitted` attribute to True after fitting the threshold.
        """

        if not self.rf_fitted:
            raise ValueError("Random forest model must be fitted first.")
        self.pipeline.steps.append(('threshold', ThresholdStep()))
        self.pipeline.steps.append(('decay', RecalculateConfidenceScores(decay_rate=get_parameter("decay_rate"), 
                                                                         midpoint=get_parameter("midpoint"), 
                                                                         cutoff=get_parameter("cutoff"))))
        y_pred = self.pipeline.named_steps['regressor'].predict(
                    self.pipeline.named_steps['pca'].transform(
                        self.pipeline.named_steps['scaler'].transform(X_calib)), quantiles=[get_parameter("quantile")])
        _ = self.pipeline.named_steps['threshold'].fit(y_pred, y_true=y_calib, y_labels=get_parameter("calibration_confidence"))
        self.threshold_fitted = True

    def fit_onn(self, X_train, y_train, ids_train):
        """
        Fit the One Nearest Neighbor model using the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).
        ids_train : DataFrame
            DataFrame containing the 'GRID_CELL' column which identifies the grid cell for each sample.
        Returns
        -------
        None
        """

        self.onn_model.fit(X_train, y_train, ids_train)
        self.onn_fitted = True

    def predict(self, X_test, y_test, ids_calib):
        """
        Predict using the fitted model and return predictions with decay and scores.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        y_test : array-like of shape (n_samples,)
            True values for X_test.
        ids_calib : DataFrame
            DataFrame containing 'ACQ_DATE' and 'GRID_CELL' columns for calibration.
        Returns
        -------
        y_pred_decay : array-like of shape (n_samples,)
            Predictions after applying decay.
        y_scores_decay : array-like of shape (n_samples,)
            Scores after applying decay.
        y_pred : array-like of shape (n_samples,)
            Raw predictions from the model.
        y_scores : array-like of shape (n_samples,)
            Raw scores from the model.
        Raises
        ------
        ValueError
            If the random forest model or threshold is not fitted.
        """

        if not self.rf_fitted or not self.threshold_fitted:
            raise ValueError("Both random forest model and threshold must be fitted first.")
        y_pred_decay, y_scores_decay, y_pred, y_scores = self.pipeline.named_steps['decay'].transform(
            self.pipeline.named_steps['threshold'].transform(
                self.pipeline.named_steps['regressor'].predict(
                    self.pipeline.named_steps['pca'].transform(
                        self.pipeline.named_steps['scaler'].transform(X_test)), 
                        quantiles=[get_parameter("quantile")]), y_test), ids_calib['ACQ_DATE'], ids_calib['GRID_CELL'])
        return y_pred_decay, y_scores_decay, y_pred, y_scores
    
    def get_onn(self, X, ids):
        """
        Get the One Nearest Neighbor model predictions for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        ids : DataFrame
            DataFrame containing the 'GRID_CELL' column which identifies the grid cell for each sample.
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
        Raises
        ------
        ValueError
            If the One Nearest Neighbor model is not fitted.
        """

        if not self.onn_fitted:
            raise ValueError("One Nearest Neighbor model must be fitted first.")
        return self.onn_model.transform(X, ids)

    @staticmethod
    def save_predictions_to_csv(fire_data, y_pred_decay, y_scores_decay, y_pred, y_scores, ids_test, file_path):
        """
        Save the prediction results to a CSV file.

        Parameters
        ----------
        fire_data : DataFrame
            The original fire data DataFrame.
        y_pred_decay : array-like of shape (n_samples,)
            Predictions after applying decay.
        y_scores_decay : array-like of shape (n_samples,)
            Scores after applying decay.
        y_pred : array-like of shape (n_samples,)
            Raw predictions from the model.
        y_scores : array-like of shape (n_samples,)
            Raw scores from the model.
        ids_test : DataFrame
            DataFrame containing 'ACQ_DATE', 'LONGITUDE', and 'LATITUDE' columns.
        file_path : str
            The path to save the prediction results CSV file.

        Returns
        -------
        None
        """
        # Create a DataFrame with the prediction results
        predictions = pd.DataFrame({
            'GRID_CELL': ids_test['GRID_CELL'],
            'ACQ_DATE': ids_test['ACQ_DATE'],
            'ABNORMAL_LABEL': y_pred,
            'SIGNIFICANCE_SCORE': y_scores,
            'ABNORMAL_LABEL_DECAY': y_pred_decay,
            'SIGNIFICANCE_SCORE_DECAY': y_scores_decay
        }) 
        # Merge fire_data with predictions to map the abnormal labels to individual fire IDs
        data = fire_data.merge(predictions[['GRID_CELL', 'ACQ_DATE', 'ABNORMAL_LABEL', 'SIGNIFICANCE_SCORE', 
                                            'ABNORMAL_LABEL_DECAY', 'SIGNIFICANCE_SCORE_DECAY']], 
                                        on=['GRID_CELL', 'ACQ_DATE'], 
                                        how='left')

        # Keep only the ACQ_DATE, LONGITUDE, LATITUDE, and prediction columns
        data = data[['ACQ_DATE', 'LONGITUDE', 'LATITUDE', 'ABNORMAL_LABEL', 'SIGNIFICANCE_SCORE', 
                     'ABNORMAL_LABEL_DECAY', 'SIGNIFICANCE_SCORE_DECAY']]

        # Save the DataFrame to a CSV file
        data.to_csv(file_path, index=False)
        print(f"Prediction results saved to {file_path}")

    @staticmethod
    def save_nn_results(X_nn_firedays, X_nn_nofiredays, y_nn_firedays, y_nn_nofiredays, X_columns, file_prefix):
        """
        Save neural network results to CSV files.
        This function converts the provided numpy arrays into pandas DataFrames,
        appends the target variable 'FIRE_COUNT_CELL' to each DataFrame, and saves
        them as CSV files.

        Parameters
        ----------
        X_nn_firedays : numpy.ndarray
            Feature matrix for days with fire incidents.
        X_nn_nofiredays : numpy.ndarray
            Feature matrix for days without fire incidents.
        y_nn_firedays : numpy.ndarray
            Target vector for days with fire incidents.
        y_nn_nofiredays : numpy.ndarray
            Target vector for days without fire incidents.
        X_columns : list of str
            List of column names for the feature matrices.
        file_prefix : str
            Path prefix for the output CSV files.
        Returns
        -------
        None
        """

        # Convert the numpy arrays to pandas DataFrames
        df_nn_firedays = pd.DataFrame(X_nn_firedays, columns=X_columns)
        df_nn_firedays['FIRE_COUNT_CELL'] = y_nn_firedays
        
        df_nn_alldays = pd.DataFrame(X_nn_nofiredays, columns=X_columns)
        df_nn_alldays['FIRE_COUNT_CELL'] = y_nn_nofiredays
        
        # Save to CSV files
        df_nn_firedays.to_csv(f"{file_prefix}_nn_results_firedays.csv", index=False)
        df_nn_alldays.to_csv(f"{file_prefix}_nn_results_alldays.csv", index=False)

    
def save(model, model_name):
    """
    Save the model to disk in multiple parts.
    This method saves the model to the directory specified by the
    `models_dir` path and `model_name`. The model is split into multiple parts, each with
    a maximum size of 90 MB.

    Parameters
    -------
    model : object
        The model to be saved.
    model_name : str
        The name of the model to be saved.
    
    Returns
    -------
    None
    """

    _ = save_large_model(model, f"{get_path("models_dir")}/{model_name}", part_size=90)

def load(model_name, parts_number=None):
    """
    Load the saved model.
    This method loads a previously saved model from the specified directory.
    The model is loaded in parts if `parts_number` is specified.

    Parameters
    ------
    model_name : str
        The name of the model to be loaded.
    parts_number : int
        The number of parts the model was split into during saving. If not provided, 
        the method will attempt to infer the number of parts based on the files in the directory.
        Defaults to None.
    """

    return load_large_model(f"{get_path("models_dir")}/{model_name}", parts_number)


def main():
    # Example usage
    start_date = pd.to_datetime('2015-01-01').date()
    end_date = pd.to_datetime('2022-02-23').date()
    calib_date = pd.to_datetime('2021-02-23').date()
    print("Loading data...")
    static_data = DataLoader.load_static_data(resolution="50km")
    fire_data, weather_data = DataLoader.load_dynamic_data(start_date=start_date, end_date=end_date)
    feature_engineering = FeatureEngineering(start_date=start_date, end_date=end_date)
    time_series_data = feature_engineering.transform(fire_data, static_data, weather_data)
    X_train, X_calib, y_train, y_calib, ids_train, ids_calib = feature_engineering.get_train_calibration_split(time_series_data, 
                                                                                                               start_date_calib=calib_date)
    print("Shape of the training data:", X_train.shape)
    print("Shape of the calibration data:", X_calib.shape)

    pipeline = FirePredictionPipeline()
    print("\nFitting the pipeline...")
    pipeline.fit_rf(X_train, y_train)
    print("Random forest model fitted successfully.")
    pipeline.fit_threshold(X_calib, y_calib)
    print("Pipeline fitted successfully.")
    print("Pipeline: ", pipeline.pipeline.steps)

    X_test, y_test, ids_test = feature_engineering.get_test_data(time_series_data[time_series_data['ACQ_DATE'] >= calib_date])
    y_pred_decay, y_scores_decay, y_pred, y_scores = pipeline.predict(X_test, y_test, ids_test)
    print("\nPredictions and scores calculated successfully.")
    pipeline.save_predictions_to_csv(fire_data[fire_data["ACQ_DATE"] >= calib_date], 
                                     y_pred_decay, y_scores_decay, y_pred, y_scores, ids_test, "results/calibration_predictions.csv")
    print("Prediction results saved successfully.")

    pipeline.fit_onn(X_train, y_train, ids_train)
    print("\nOne Nearest Neighbor model fitted successfully.")
    X_nn_firedays, X_nn_nofiredays, y_nn_firedays, y_nn_nofiredays = pipeline.get_onn(X_test, ids_test)
    print("Nearest neighbors calculated successfully.")
    pipeline.save_nn_results(X_nn_firedays, X_nn_nofiredays, y_nn_firedays, y_nn_nofiredays, X_train.columns, 
                             "results/calibration")
    print("Nearest neighbors results saved successfully.")

    save(pipeline, "pipeline")
    print("\nPipeline saved successfully.")
    pipeline_loaded = load("pipeline")
    print("Pipeline loaded successfully.")
    print("Pipeline: ", pipeline_loaded.pipeline.steps)

    # Test the model on a new dataset
    start_date = pd.to_datetime('2022-02-24').date()
    end_date = pd.to_datetime('2024-09-30').date()
    print("\nLoading test data...")
    static_data = DataLoader.load_static_data(resolution="50km")
    fire_data, weather_data = DataLoader.load_dynamic_data(start_date=start_date, end_date=end_date)
    feature_engineering = FeatureEngineering(start_date=start_date, end_date=end_date)
    time_series_data = feature_engineering.transform(fire_data, static_data, weather_data)
    X_test, y_test, ids_test = feature_engineering.get_test_data(time_series_data)
    print("Shape of the test data:", X_test.shape)

    y_pred_decay, y_scores_decay, y_pred, y_scores = pipeline.predict(X_test, y_test, ids_test)
    print("Predictions and scores calculated successfully.")
    pipeline.save_predictions_to_csv(fire_data, y_pred_decay, y_scores_decay, y_pred, y_scores, ids_test, 
                                     "results/test_predictions.csv")
    print("Prediction results saved successfully.")
    
    X_nn_firedays, X_nn_nofiredays, y_nn_firedays, y_nn_nofiredays = pipeline.get_onn(X_test, ids_test)
    print("Nearest neighbors calculated successfully.")
    pipeline.save_nn_results(X_nn_firedays, X_nn_nofiredays, y_nn_firedays, y_nn_nofiredays, X_train.columns, 
                             "results/calibration")
    print("Nearest neighbors results saved successfully.")

if __name__ == "__main__":
    main()