import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


# -------------------------------------------------
# Question 1: Dataset generation and visualization
# -------------------------------------------------

def generate_clean_data(
    n_samples=500,
    noise=20,
    random_state=42
):
    """
    Generate a clean synthetic regression dataset.

    Return:
        X, y, true_coef
    """
    X, y, coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=random_state
    )

    return X, y, float(coef)


def add_outliers(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Add artificial outliers to the first n_outliers observations.
    """
    rng = np.random.RandomState(random_state)

    X_out = X.copy()
    y_out = y.copy()

    X_out[:n_outliers] = 10 + 0.75 * rng.randn(n_outliers, 1)
    y_out[:n_outliers] = -15 + 20 * rng.randn(n_outliers)

    return X_out, y_out


def plot_dataset_with_outliers(
    X,
    y,
    n_outliers=25
):
    """
    Plot dataset highlighting outliers.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        X[n_outliers:],
        y[n_outliers:],
        label="Normal Data"
    )

    ax.scatter(
        X[:n_outliers],
        y[:n_outliers],
        label="Artificial Outliers"
    )

    ax.set_title("Dataset with Artificial Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


# -------------------------------------------------
# Question 2: Fit regression models
# -------------------------------------------------

def fit_linear_regression(X, y):
    """
    Fit ordinary Linear Regression.
    """
    model = LinearRegression()
    model.fit(X, y)

    return float(model.coef_[0])


def fit_huber_regression(X, y):
    """
    Fit Huber Regression.
    """
    model = HuberRegressor()
    model.fit(X, y)

    return float(model.coef_[0])


def fit_ransac_regression(X, y, random_state=42):
    """
    Fit RANSAC Regression.
    """
    model = RANSACRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    return float(model.estimator_.coef_[0])


def fit_theilsen_regression(X, y, random_state=42):
    """
    Fit Theil-Sen Regression.
    """
    model = TheilSenRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    return float(model.coef_[0])


def coefficient_errors(coef_dict, true_coef):
    """
    Compute absolute coefficient errors.
    """
    return {
        name: abs(coef - true_coef)
        for name, coef in coef_dict.items()
    }


def best_robust_model(errors):
    """
    Return robust model with smallest error.
    """
    robust_models = {
        k: v
        for k, v in errors.items()
        if k in (
            "huber_regression",
            "ransac_regression",
            "theilsen_regression"
        )
    }

    return min(robust_models, key=robust_models.get)


def ransac_outlier_summary(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Return:
        total_outliers_detected,
        added_outliers_detected
    """
    model = RANSACRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    total_outliers_detected = int(np.sum(outlier_mask))

    added_outliers_detected = int(
        np.sum(outlier_mask[:n_outliers])
    )

    return (
        total_outliers_detected,
        added_outliers_detected
    )


# -------------------------------------------------
# Question 2: Visualization functions
# -------------------------------------------------

def plot_regression_fits(
    X,
    y,
    random_state=42
):
    """
    Plot fitted regression lines.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X, y, alpha=0.6, label="Data")

    x_line = np.linspace(
        X.min(),
        X.max(),
        500
    ).reshape(-1, 1)

    linear = LinearRegression()
    linear.fit(X, y)

    huber = HuberRegressor()
    huber.fit(X, y)

    ransac = RANSACRegressor(
        random_state=random_state
    )
    ransac.fit(X, y)

    theilsen = TheilSenRegressor(
        random_state=random_state
    )
    theilsen.fit(X, y)

    ax.plot(
        x_line,
        linear.predict(x_line),
        label="Linear Regression"
    )

    ax.plot(
        x_line,
        huber.predict(x_line),
        label="Huber Regression"
    )

    ax.plot(
        x_line,
        ransac.predict(x_line),
        label="RANSAC Regression"
    )

    ax.plot(
        x_line,
        theilsen.predict(x_line),
        label="Theil-Sen Regression"
    )

    ax.set_title("Regression Model Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


def plot_ransac_inliers_outliers(
    X,
    y,
    random_state=42
):
    """
    Plot RANSAC inliers vs outliers.
    """
    model = RANSACRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        X[inlier_mask],
        y[inlier_mask],
        label="Inliers"
    )

    ax.scatter(
        X[outlier_mask],
        y[outlier_mask],
        label="Outliers"
    )

    ax.set_title("RANSAC Inliers and Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig
