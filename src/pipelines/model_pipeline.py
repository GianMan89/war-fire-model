import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from quantile_forest import RandomForestQuantileRegressor
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

# Add the project root to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '../../'))  # Adjust based on depth
sys.path.append(project_root)

from config.config import get_parameter
from utils.file_utils import get_path, save_large_model, load_large_model
from src.models.error_threshold import ThresholdStep
from src.models.score_decay import RecalculateConfidenceScores
from src.models.nearest_neighbor import OneNearestNeighborModel
from src.models.explainer import LimeExplainer
from src.pipelines.data_pipeline import DataPipeline

class FirePredictionPipeline:
    """
    A pipeline for fire prediction using a combination of scaling, PCA, and a RandomForestQuantileRegressor.
    This pipeline also includes a threshold step and a decay step for recalculating confidence scores.
    The pipeline also includes a OneNearestNeighborModel for finding the nearest neighbors for each sample 
    and a LimeExplainer for generating explanations.

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
    explainer_fitted : bool
        Indicates whether the explainer model has been fitted.
    Methods
    -------
    fit_rf(X_train, y_train)
        Fits the random forest model using the training data.
    fit_threshold(X_calib, y_calib)
        Fits the threshold model using the calibration data. Requires the random forest model to be fitted first.
    fit_onn(X_train, y_train, ids_train)
        Fits the One Nearest Neighbor model using the training data.
    fit_explainer(X_train, feature_names)
        Fits the explainer model using the training data.
    fit(X_train, y_train, X_calib, y_calib, ids_train)
        Fits the pipeline using the training and calibration data.
    predict(X_test, y_test, ids_calib)
        Predicts the fire occurrence and confidence scores using the test data and calibration IDs.
    get_onn(X, ids)
        Gets the nearest neighbors for the input data using the One Nearest Neighbor model.
    get_explanation(instance, num_features=16)
        Gets an explanation for a single instance using the LIME explainer.
    save_explanation_instance(exp, file_name="explanation")
        Saves the explanation as an HTML file.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=get_parameter("n_components"), random_state=get_parameter("random_state"))),
            ('regressor', RandomForestQuantileRegressor(n_estimators=get_parameter("n_estimators"), random_state=get_parameter("random_state"))),
        ])
        self.onn_model = OneNearestNeighborModel()
        self.explainer = LimeExplainer()
        self.rf_fitted = False
        self.threshold_fitted = False
        self.onn_fitted = False
        self.explainer_fitted = False

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

    def fit_explainer(self, X_train, feature_names):
        """
        Fit the explainer model using the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        feature_names : list
            The names of the features in the training data.
        Returns
        -------
        None
        """

        self.explainer.fit(X_train, feature_names)
        self.explainer_fitted = True

    def fit(self, X_train, y_train, X_calib, y_calib, ids_train):
        """
        Fit the pipeline using the training and calibration data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values for the training data.
        X_calib : array-like of shape (n_samples, n_features)
            The calibration input samples.
        y_calib : array-like of shape (n_samples,)
            The target values for the calibration data.
        ids_train : DataFrame
            DataFrame containing 'ACQ_DATE' and 'GRID_CELL' columns for training.
        Returns
        -------
        None
        """

        self.fit_rf(X_train, y_train)
        self.fit_threshold(X_calib, y_calib)
        self.fit_onn(X_train, y_train, ids_train)
        self.fit_explainer(X_train, X_train.columns)
        return self

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
    
    def get_explanation(self, instance, num_features=16):
        """
        Get an explanation for a single instance using the LIME explainer.

        Parameters
        ----------
        instance : np.ndarray
            The instance to explain.
        num_features : int
            The number of features to include in the explanation.
        Returns
        -------
        exp : lime.explanation.Explanation
            The explanation object.
        """
        predict_fn = lambda x: self.pipeline.named_steps['regressor'].predict(
            self.pipeline.named_steps['pca'].transform(
                self.pipeline.named_steps['scaler'].transform(x)), quantiles=[0.95])
        return self.explainer.explain_instance(instance, predict_fn, num_features)
    
    def save_explanation_instance(self, exp, file_name="explanation"):
        """
        Save the explanation as an HTML file.

        Parameters
        ----------
        exp : lime.explanation.Explanation
            The explanation object.
        file_name : str
            The name of the file to save the explanation to.
        Returns
        -------
        None
        """

        self.explainer.save_explanation_instance(exp, f"{file_name}.html")

    def explain_data(self, data, num_features=16, file_path=None):
        """
        Explain a dataset using the LIME explainer.

        Parameters
        ----------
        data : np.ndarray
            The data to explain.
        num_features : int
            The number of features to include in the explanation.
        file_path : str, optional
            The path to save the explanation as a csv file (default is None).
        Returns
        -------
        explanations_df : pd.DataFrame
            The explanations for the data.
        """

        predict_fn = lambda x: self.pipeline.named_steps['regressor'].predict(
            self.pipeline.named_steps['pca'].transform(
                self.pipeline.named_steps['scaler'].transform(x)), quantiles=[0.95])
        
        return self.explainer.explain_data(data, predict_fn, num_features, file_path)

    @staticmethod
    def save_predictions_to_csv(y_pred, fire_data, ids_test, file_path):
        """
        Save the prediction results to a CSV file.

        Parameters
        ----------
        y_pred : tuple
            Tuple containing the predictions and scores.
        fire_data : DataFrame
            The original fire data DataFrame.
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
            'ABNORMAL_LABEL': y_pred[2],
            'SIGNIFICANCE_SCORE': y_pred[3],
            'ABNORMAL_LABEL_DECAY': y_pred[0],
            'SIGNIFICANCE_SCORE_DECAY': y_pred[1]
        }) 
        # Merge fire_data with predictions to map the abnormal labels to individual fire IDs
        data = fire_data.merge(predictions[['GRID_CELL', 'ACQ_DATE', 'ABNORMAL_LABEL', 'SIGNIFICANCE_SCORE', 
                                            'ABNORMAL_LABEL_DECAY', 'SIGNIFICANCE_SCORE_DECAY']], 
                                        on=['GRID_CELL', 'ACQ_DATE'], 
                                        how='left')
        # Keep only the ACQ_DATE, LONGITUDE, LATITUDE, and prediction columns
        data = data[['ACQ_DATE', 'LONGITUDE_ORIGINAL', 'LATITUDE_ORIGINAL', 'ABNORMAL_LABEL', 'SIGNIFICANCE_SCORE', 
                     'ABNORMAL_LABEL_DECAY', 'SIGNIFICANCE_SCORE_DECAY']]
        # Rename labels
        data.rename(columns={'LATITUDE_ORIGINAL': 'LATITUDE', 'LONGITUDE_ORIGINAL': 'LONGITUDE'}, inplace=True)
        # Save the DataFrame to a CSV file
        data.to_csv(file_path, index=False)
        print(f"Prediction results saved to {file_path}")

    @staticmethod
    def save_nn_results(nn_results, X_test, y_test, file_prefix):
        """
        Save neural network results to CSV files.
        This function converts the provided numpy arrays into pandas DataFrames,
        appends the target variable 'FIRE_COUNT_CELL' to each DataFrame, and saves
        them as CSV files.

        Parameters
        ----------
        nn_results : tuple
            Tuple containing the results for fire days and no fire days.
        X_test : DataFrame
            The test data used for prediction.
        y_test : DataFrame
            The target values for the test data.
        file_prefix : str
            Path prefix for the output CSV files.
        Returns
        -------
        None
        """

        # Convert the numpy arrays to pandas DataFrames
        df_nn_firedays = pd.DataFrame(nn_results[0], columns=X_test.columns)
        df_nn_firedays['FIRE_COUNT_CELL'] = nn_results[2] 
        df_nn_nofiredays = pd.DataFrame(nn_results[1], columns=X_test.columns)
        df_nn_nofiredays['FIRE_COUNT_CELL'] = nn_results[3]
        df_original = X_test.copy()
        df_original['FIRE_COUNT_CELL'] = y_test
        # Save to CSV files
        df_nn_firedays.to_csv(f"{file_prefix}_nn_results_firedays.csv", index=False)
        df_nn_nofiredays.to_csv(f"{file_prefix}_nn_results_nofiredays.csv", index=False)
        df_original.to_csv(f"{file_prefix}_original_data.csv", index=False)

    
def save_pipeline(model, model_name, resolution="50km"):
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
    resolution : str
        The resolution used. Default is "50km".
    
    Returns
    -------
    None
    """

    _ = save_large_model(model, f"{get_path("models_dir")}/{resolution}/{model_name}", part_size=90)

def load_pipeline(model_name, parts_number=None, resolution="50km"):
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
    resolution : str
        The resolution used. Default is "50km".
    """

    return load_large_model(f"{get_path("models_dir")}/{resolution}/{model_name}", parts_number)


if __name__ == "__main__":
    resolution = "50km"
    start_date_train = pd.to_datetime('2015-01-01').date()
    end_date_train = pd.to_datetime('2021-02-23').date()
    start_date_calib = pd.to_datetime('2021-02-24').date()
    end_date_calib = pd.to_datetime('2022-02-23').date()
    start_date_test = pd.to_datetime('2022-02-24').date()
    end_date_test = pd.to_datetime('2024-09-30').date()

    data_pipeline = DataPipeline(resolution=resolution)
    X_train, y_train, ids_train, fires_train = data_pipeline.fit_transform(start_date=start_date_train, end_date=end_date_train)
    X_calib, y_calib, ids_calib, fires_calib = data_pipeline.transform(start_date=start_date_calib, end_date=end_date_calib)
    X_test, y_test, ids_test, fires_test = data_pipeline.transform(start_date=start_date_test, end_date=end_date_test)
    print("Data loaded successfully.")
    print("X_train shape:", X_train.shape)
    print("X_calib shape:", X_calib.shape)
    print("X_test shape:", X_test.shape)

    save_large_model(data_pipeline, f"{get_path('models_dir')}/{resolution}/data_pipeline", part_size=90)
    print("Data pipeline saved successfully!")
    data_pipeline_loaded = load_large_model(f"{get_path('models_dir')}/{resolution}/data_pipeline")
    print("Data pipeline loaded successfully!")

    model_pipeline = FirePredictionPipeline()
    print("Fitting the model pipeline...")
    model_pipeline.fit(X_train, y_train, X_calib, y_calib, ids_train)
    print("Model pipeline fitted successfully.")
    print("Model pipeline: ", model_pipeline.pipeline.steps)

    save_pipeline(model_pipeline, "model_pipeline", resolution=resolution)
    print("Model pipeline saved successfully.")
    model_pipeline_loaded = load_pipeline("model_pipeline", resolution=resolution)
    print("Model pipeline loaded successfully.")
    print("Model pipeline: ", model_pipeline_loaded.pipeline.steps)

    y_pred = model_pipeline_loaded.predict(X_test, y_test, ids_test)
    print("Test predictions and scores calculated successfully.")
    model_pipeline_loaded.save_predictions_to_csv(y_pred, fires_test, ids_test, f"results/{resolution}/test_predictions.csv")
    print("Test prediction results saved successfully.")
    
    onn_results = model_pipeline_loaded.get_onn(X_test.iloc[:400000], ids_test.iloc[:400000])
    print("Nearest neighbors calculated successfully.")
    model_pipeline_loaded.save_nn_results(onn_results, X_test, y_test, f"results/{resolution}/test")
    print("Nearest neighbors results saved successfully.")

    idx_instance = 2000
    instance = X_test.iloc[idx_instance]
    exp = model_pipeline_loaded.get_explanation(instance.values)
    model_pipeline_loaded.save_explanation_instance(exp, f"results/{resolution}/plots/explanation_{idx_instance}")
    print("Explanation instance generated successfully.")

    # Explain the test data
    X_test_subset = X_test.iloc[:1000].reset_index(drop=True)
    explanations_df = model_pipeline_loaded.explain_data(X_test_subset, file_path=f"results/{resolution}/test_explanations.csv")
    print("Explanations generated successfully.")