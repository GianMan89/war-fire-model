import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

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