import os
import sys
import numpy as np
import lime.lime_tabular
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

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
        """
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(training_data),
            feature_names=feature_names,
            class_names=self.class_names,
            mode=self.mode
        )

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
        """
        exp = self.explainer.explain_instance(np.array(instance), 
                                              predict_fn, num_features=num_features)
        return exp

    def save_explanation(self, exp, file_name):
        """
        Save the explanation as an HTML file.

        Parameters
        ----------
        exp : lime.explanation.Explanation
            The explanation object.
        """
        with open(file_name, 'w') as f:
            f.write(exp.as_html())
        plt.show()
