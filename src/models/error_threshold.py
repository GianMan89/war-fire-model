import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings("ignore")

# Add the project root to the Python path
try:
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
    sys.path.append(project_root)
except Exception as e:
    raise RuntimeError(f"Failed to add project root to Python path: {e}")

class ThresholdStep(BaseEstimator, TransformerMixin):
    """
    ThresholdStep is a custom transformer that determines the best threshold for classification based on the provided predictions and true values.
    
    Attributes
    ----------
    threshold : float, optional
        Initial threshold value. If not provided, it will be determined during the fitting process.
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

        # Check if y_true and y_labels are provided
        if y_true is None or y_labels is None:
            raise ValueError('Transformer requires y_true and y_labels to be passed.')
        
        # Calculate the error between true and predicted values, set negative errors to zero
        error = y_true - y_pred
        error[error < 0] = 0

        # If y_labels is a float, determine threshold based on a fraction of total fires
        if isinstance(y_labels, float):
            try:
                sorted_indices = np.argsort(error).reset_index(drop=True)
                sorted_y_val = y_true.iloc[sorted_indices]
                cumulative_fires = np.cumsum(sorted_y_val)
                total_fires = np.sum(y_true)
                threshold_index = np.searchsorted(cumulative_fires, y_labels * total_fires)
                threshold = error.iloc[sorted_indices[threshold_index]]
                y_labels = (error > threshold).astype(int)
                self.parts_number = None
            except Exception as e:
                raise ValueError(f"An error occurred while calculating threshold from fraction: {e}")
        
        # Initialize variables to store the best threshold and highest balanced accuracy
        best_threshold = None
        highest_balanced_accuracy = 0
        thresholds = np.linspace(error.min(), error.max(), 1000)

        # Iterate through thresholds to determine the one with the highest balanced accuracy
        for threshold in thresholds:
            try:
                y_pred_threshold = (error > threshold).astype(int)
                balanced_acc = balanced_accuracy_score(y_labels, y_pred_threshold)
                if balanced_acc > highest_balanced_accuracy:
                    highest_balanced_accuracy = balanced_acc
                    best_threshold = threshold
            except Exception as e:
                # Skip thresholds that raise exceptions
                warnings.warn(f"An error occurred while evaluating threshold {threshold}: {e}")
                continue

        # Set the best threshold found
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

        # Check if threshold is set
        if self.threshold is None:
            raise ValueError("Threshold is not set. Please fit the transformer first.")
        
        # Check if true labels are provided
        if y_true is None:
            raise ValueError("True labels are required for the transformation.")
        
        # Calculate the error between true and predicted values, set negative errors to zero
        error = y_true - y_pred
        error[error < 0] = 0
        
        # Determine if the error is abnormal
        is_abnormal = error > self.threshold
        
        # Calculate significance score for each prediction
        try:
            significance_score = np.array(pd.Series(error).apply(lambda value: ((value - self.threshold) / self.threshold) if value < self.threshold else (value - self.threshold) / value))
        except Exception as e:
            raise ValueError(f"An error occurred while calculating significance scores: {e}")
        
        # Return transformed output
        return np.array([is_abnormal.astype(int), significance_score])