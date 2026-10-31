
import numpy as np

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV, KFold

from feature_engineering import engineer_features


def make_preprocessor(data):
    """Create preprocessing pipelines for numerical and categorical columns."""

    numeric_features = data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = data.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])


def make_model_pipeline(model, training_features, log_target=False):
    """Build a complete feature engineering and preprocessing pipeline."""

    pipeline = Pipeline([
        (
            "feature_engineering",
            FunctionTransformer(engineer_features, validate=False)
        ),
        ("preprocessor", make_preprocessor(training_features)),
        ("model", model)
    ])

    if log_target:
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )

    return pipeline


def get_baseline_models():
    """Return the baseline regression models."""

    return {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=0.1)": Ridge(alpha=0.1),
        "Log-target Linear Regression": LinearRegression()
    }


def get_random_forest_parameters():
    """Return the Random Forest hyperparameter search space."""

    return {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.7, 1.0]
    }


def get_xgboost_parameters():
    """Return the XGBoost hyperparameter search space."""

    return {
        "model__n_estimators": [200, 300, 400, 600],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [1, 5, 10]
    }


def tune_model(pipeline, parameters, X_train, y_train, cv, n_iter=20):
    """Tune a model using RandomizedSearchCV and RMSE."""

    scoring = {
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=n_iter,
        scoring=scoring,
        refit="RMSE",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    search.fit(X_train, y_train)

    return search
