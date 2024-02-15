import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_curve, auc


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_fields=["year", "month", "day", "weekday", "quarter"]):
        self.date_fields = date_fields

    def fit(self, X, y=None):
        self.feature_names_out = []
        for date_column in X.columns:
            self.feature_names_out.extend(
                [f"{date_column}_{date_field}" for date_field in self.date_fields]
            )

        return self

    def get_feature_names_out(self):
        return self.feature_names_out

    def transform(self, X):
        for date_column in X.columns:
            # Ensure the column exists in the DataFrame
            if date_column not in X.columns:
                raise ValueError(f"{date_column} not found in the DataFrame.")

            # Convert the specified column to datetime
            X[date_column] = pd.to_datetime(X[date_column])

            # Extract date components
            for date_field in self.date_fields:
                X[f"{date_column}_{date_field}"] = getattr(
                    X[date_column].dt, date_field
                )

            # Drop the original date column
            X = X.drop(columns=[date_column])

        return X


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, z_threshold=3):
        self.z_threshold = z_threshold

    def fit(self, X, y=None):
        self.feature_names_out = []
        for column_name in X.columns:
            self.feature_names_out.append(column_name)

        return self

    def get_feature_names_out(self):
        return self.feature_names_out

    def transform(self, X):
        for column_name in X.columns:
            mean = X[column_name].mean()
            std = X[column_name].std()
            z_scores = np.abs((X[column_name] - mean) / std)
            max_value = X[z_scores < self.z_threshold][column_name].max()
            X.loc[X[column_name] > max_value, column_name] = max_value

        return X


def plot_roc_curve(model, X_test, y_test):
    y_pred_probs = model.predict_proba(X_test)

    # Compute ROC curve and area under the curve (AUC)
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(5, 5))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


import numpy as np


def iqr_filter(data, multiplier=1.5):
    """
    Filter outliers from the input data using the Interquartile Range (IQR) method.

    Parameters:
        data (array-like): Input data array.
        multiplier (float): Multiplier to determine the range for outlier detection.
                            Defaults to 1.5, but can be adjusted as needed.

    Returns:
        numpy.ndarray: Filter mask with outliers removed.
    """
    # Calculate the first and third quartiles
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # Calculate the IQR (Interquartile Range)
    iqr = q3 - q1

    # Define the lower and upper bounds for outlier detection
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    # Filter out the outliers from the original dataset
    return (data >= lower_bound) & (data <= upper_bound)


def zscore_filter(data, threshold=3, nan=0):
    """
    Filter outliers from the input data using the Z-score method.

    Parameters:
        data (array-like): Input data array.
        threshold (float): Threshold value for Z-score-based outlier detection.
                           Data points with absolute Z-scores greater than this threshold
                           are considered outliers. Defaults to 3.

    Returns:
        numpy.ndarray: Filter mask with outliers removed.
    """
    # Calculate the Z-scores for each data point
    z_scores = stats.zscore(np.nan_to_num(data, nan=nan))

    # Filter out the outliers from the original dataset
    return np.abs(z_scores) < threshold


def plot_feature_importances(feature_names, feature_importances, n=10):
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importances}
    ).sort_values("importance", ascending=False)

    sns.barplot(x="importance", y="feature", hue="feature", data=importance_df.head(n))
    plt.show()

    return importance_df
