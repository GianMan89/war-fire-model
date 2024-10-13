import os
import sys
import pandas as pd
import numpy as np
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

        Raises
        ------
        OverflowError
            If an overflow occurs during the exponential calculation.
        """
        try:
            if time_diff > self.cutoff:
                return 0
            return 1 / (1 + np.exp(self.decay_rate * (time_diff - self.midpoint)))
        except OverflowError as e:
            # Handle potential overflow in the exponential calculation
            print(f"OverflowError encountered in sigmoid_decay: {e}")
            return 0

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
        # Validate input parameters
        if dates is None or grid_cells is None:
            raise ValueError("Dates and grid cells are required for the recalculation of confidence scores.")

        try:
            # Extract original confidence scores and predicted labels
            y_scores = X[1]
            y_pred = X[0]
            
            # Create a list of indexed events to track their original position after sorting
            indexed_events = list(enumerate(zip(y_scores, dates, grid_cells)))
            
            # Sort events by date to ensure the temporal relationship is preserved
            indexed_events_sorted = sorted(indexed_events, key=lambda x: x[1][1])
            
            # Initialize a list to store recalculated confidence scores
            recalculated_confidences = [None] * len(y_scores)
            
            # Track the last significant (war-related) event for each grid cell
            last_war_events = {}
            
            # Iterate over events in chronological order to recalculate confidence scores
            for i, (original_index, (current_conf, current_date, grid_cell)) in enumerate(indexed_events_sorted):
                # If the current event is significant (war-related), update the last war event for the grid cell
                if current_conf > 0:
                    last_war_events[grid_cell] = {
                        'ACQ_DATE': current_date,
                        'SIGNIFICANCE_SCORE': current_conf
                    }
                    recalculated_confidences[original_index] = current_conf
                # If the current event is not significant, apply decay based on the last war event in the same grid cell
                elif grid_cell in last_war_events:
                    last_war_event = last_war_events[grid_cell]
                    current_date = pd.to_datetime(current_date)
                    last_war_event['ACQ_DATE'] = pd.to_datetime(last_war_event['ACQ_DATE'])
                    
                    # Calculate the time difference in days between the current event and the last significant event
                    time_diff = (current_date - last_war_event['ACQ_DATE']) / np.timedelta64(1, 'D')
                    
                    # Apply the sigmoid decay function to determine the influence of the last war event
                    decayed_influence = self.sigmoid_decay(time_diff) * last_war_event['SIGNIFICANCE_SCORE']
                    
                    # Update the confidence score if the decayed influence is higher than the current score
                    if decayed_influence > current_conf and decayed_influence > 0:
                        new_conf = decayed_influence
                    else:
                        new_conf = current_conf
                    recalculated_confidences[original_index] = new_conf
                else:
                    # If there is no previous significant event, retain the original confidence score
                    recalculated_confidences[original_index] = current_conf
            
            # Convert recalculated confidence scores to numpy array
            recalculated_confidences = np.array(recalculated_confidences)
            
            # Recalculate binary labels based on the new confidence scores
            labels = np.where(recalculated_confidences > 0, 1, 0)
            
            return [labels, recalculated_confidences, y_pred, y_scores]
        except Exception as e:
            # Handle unexpected errors during the transformation process
            print(f"An error occurred during transformation: {e}")
            raise

# Example usage of the transformer
if __name__ == "__main__":
    # Example input data
    y_pred = np.array([1, 0, 1, 0, 0])
    y_scores = np.array([0.8, -0.5, 0.9, -0.2, -0.7])
    dates = np.array(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
    grid_cells = np.array(['A', 'A', 'B', 'B', 'A'])

    # Instantiate the transformer
    transformer = RecalculateConfidenceScores(decay_rate=0.1, midpoint=5, cutoff=10)

    # Transform the data
    try:
        transformed_data = transformer.transform([y_pred, y_scores], dates=dates, grid_cells=grid_cells)
        print("Transformed Data:", transformed_data)
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")