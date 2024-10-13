import os
import sys
import numpy as np
import lime.lime_tabular
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Add the project root to the Python path
try:
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
    sys.path.append(project_root)
except Exception as e:
    raise RuntimeError(f"Failed to add project root to Python path: {e}")

class LimeExplainer:
    """ 
    A class to create and use a LIME (Local Interpretable Model-agnostic Explanations) explainer.

    Attributes
    ----------
    training_data : np.ndarray
        The training data used to fit the explainer.
    feature_names : list
        The names of the features in the training data.
    class_names : list
        The names of the classes for classification or ['Normal', 'Abnormal'] for regression.
    mode : str
        The mode of the explainer, either 'classification' or 'regression'.
    explainer : lime.lime_tabular.LimeTabularExplainer
        The LIME explainer object.

    Methods
    -------
    explain_instance(instance, predict_fn, num_features):
        Explains a single instance using the LIME explainer.
    show_explanation(exp):
        Displays the explanation in the notebook and as a plot.
    """

    def __init__(self, class_names=['Normal', 'Abnormal'], mode='regression'):
        """
        Initializes the LimeExplainer object.

        Parameters
        ----------
        class_names : list, optional
            The names of the classes for classification or ['Normal', 'Abnormal'] for regression (default is ['Normal', 'Abnormal']).
        mode : str, optional
            The mode of the explainer, either 'classification' or 'regression' (default is 'regression').
        """
        # Set the class names and mode
        self.class_names = class_names
        self.mode = mode
        self.explainer = None

    def fit(self, training_data, feature_names):
        """
        Fits the explainer to the training data.

        Parameters
        ----------
        training_data : np.ndarray
            The training data used to fit the explainer.
        feature_names : list
            The names of the features in the training data.

        Raises
        ------
        ValueError
            If training_data or feature_names are invalid.
        """
        try:
            # Ensure that training_data is in the right format
            if not isinstance(training_data, (np.ndarray, pd.DataFrame)):
                raise ValueError("Training data must be a numpy array or pandas DataFrame.")
            if not isinstance(feature_names, list):
                raise ValueError("Feature names must be a list.")
            
            # Convert training_data to numpy array if it's a DataFrame
            training_data = np.array(training_data)
            
            # Initialize the LIME explainer
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                class_names=self.class_names,
                mode=self.mode
            )
        except Exception as e:
            print(f"An error occurred while fitting the explainer: {e}")
            raise

    def explain_instance(self, instance, predict_fn, num_features):
        """
        Explains a single instance using the LIME explainer.

        Parameters
        ----------
        instance : np.ndarray
            The instance to explain.
        predict_fn : function
            The prediction function of the model.
        num_features : int
            The number of features to include in the explanation.

        Returns
        -------
        exp : lime.explanation.Explanation
            The explanation object.

        Raises
        ------
        ValueError
            If instance, predict_fn, or num_features are invalid.
        """
        try:
            # Ensure the instance is a numpy array
            if not isinstance(instance, (np.ndarray, pd.Series)):
                raise ValueError("Instance must be a numpy array or pandas Series.")
            # Ensure num_features is an integer and greater than 0
            if not isinstance(num_features, int) or num_features <= 0:
                raise ValueError("num_features must be a positive integer.")
            # Ensure predict_fn is callable
            if not callable(predict_fn):
                raise ValueError("predict_fn must be a callable function.")
            
            # Convert instance to numpy array if it's a pandas Series
            instance = np.array(instance)
            
            # Explain the instance using the LIME explainer
            exp = self.explainer.explain_instance(instance, predict_fn, num_features=num_features)
            return exp
        except Exception as e:
            print(f"An error occurred while explaining the instance: {e}")
            raise

    def save_explanation_instance(self, exp, file_name):
        """
        Save the explanation as an HTML file.

        Parameters
        ----------
        exp : lime.explanation.Explanation
            The explanation object.
        file_name : str
            The file path to save the explanation HTML file.

        Raises
        ------
        ValueError
            If exp or file_name are invalid.
        """
        try:
            # Ensure exp is a LIME explanation object
            if not isinstance(exp, lime.explanation.Explanation):
                raise ValueError("exp must be a lime.explanation.Explanation object.")
            # Ensure file_name is a string
            if not isinstance(file_name, str):
                raise ValueError("file_name must be a string.")
            
            # Save the explanation to an HTML file
            exp.save_to_file(file_name, labels=None, predict_proba=False, show_predicted_value=True)
        except Exception as e:
            print(f"An error occurred while saving the explanation: {e}")
            raise

    def explain_data(self, data, predict_fn, num_features, file_path=None):
        """
        Explains a dataset using the LIME explainer.

        Parameters
        ----------
        data : np.ndarray
            The data to explain.
        predict_fn : function
            The prediction function of the model.
        num_features : int
            The number of features to include in the explanation.
        file_path : str, optional
            The path to save the explanation as a CSV file (default is None).
        
        Returns
        -------
        explanations_df : pd.DataFrame
            The explanations for the data.

        Raises
        ------
        ValueError
            If data, predict_fn, or num_features are invalid.
        """
        try:
            # Ensure data is a pandas DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError("data must be a pandas DataFrame.")
            # Ensure num_features is an integer and greater than 0
            if not isinstance(num_features, int) or num_features <= 0:
                raise ValueError("num_features must be a positive integer.")
            # Ensure predict_fn is callable
            if not callable(predict_fn):
                raise ValueError("predict_fn must be a callable function.")
            
            explanations = []
            # Iterate over each row to explain the instance
            for i in range(data.shape[0]):
                exp = self.explain_instance(data.iloc[i], predict_fn, num_features)
                explanations.append(exp.as_list())

            # Create a DataFrame with explanations
            columns = [f'Explanation_{i}' for i in range(num_features)]
            explanations_df = pd.DataFrame(explanations, columns=columns)

            # Save explanations DataFrame to a CSV file if a path is provided
            if file_path:
                explanations_df.to_csv(file_path, index=False)
                
            return explanations_df
        except Exception as e:
            print(f"An error occurred while explaining the data: {e}")
            raise