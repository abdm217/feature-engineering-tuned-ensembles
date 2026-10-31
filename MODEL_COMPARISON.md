
# Feature Engineering & Tuned Ensembles for Regression

## 1. Project Overview

The objective of this project was to improve the house price prediction models developed in the previous regression task.

The work included:

- Creating at least four meaningful domain-based features.
- Re-evaluating baseline models using 5-fold cross-validation.
- Tuning Random Forest and XGBoost using RandomizedSearchCV.
- Comparing model performance, stability, and training time.
- Evaluating the final selected models on the untouched 20% holdout test set.

Dataset: Ames Housing Dataset

Target variable: SalePrice

Original dataset shape: 1460 rows and 86 columns.

The target, identifier column (`Id`), and five outlier indicator columns were excluded from the input features.

Final input features before engineering: 79.

Train/test split: 80% training and 20% testing.

Random state: 42.

Training samples: 1168.

Testing samples: 292.

## 2. Feature Engineering

Five new features were created using existing housing attributes.

| Engineered Feature | Description | Reason for Inclusion |
|---|---|---|
| TotalSF | Total basement, first-floor, and second-floor area | Represents the overall size of the property. |
| HouseAge | Year sold minus year built | Captures the age of the house at the time of sale. |
| TotalBathrooms | Full bathrooms plus half bathrooms counted as 0.5 | Represents the property's bathroom capacity. |
| HasRemodeled | Indicates whether the remodeling year is later than the construction year | Captures whether the property has been updated. |
| TotalPorchSF | Combined area of open porch, enclosed porch, three-season porch, screen porch, and wood deck | Represents additional outdoor living space. |

The dataset increased from 79 to 84 input features after feature engineering.

Missing area and bathroom values used in the engineered calculations were treated as zero where appropriate.

The feature engineering function was placed inside the modeling pipeline using FunctionTransformer. This ensured that feature creation was consistently applied during cross-validation and prediction.

## 3. Preprocessing

The following preprocessing steps were used:

### Numerical Features

- Missing values were replaced using median imputation.
- StandardScaler was used for feature scaling.

### Categorical Features

- Missing values were replaced using the most frequent value.
- OneHotEncoder was used to convert categorical values into numerical features.
- Unknown categories were handled using `handle_unknown="ignore"`.

A ColumnTransformer combined the numerical and categorical preprocessing pipelines.

All preprocessing operations were fitted within the training folds to avoid data leakage.

## 4. Five-Fold Cross-Validation

The baseline models were evaluated using five-fold cross-validation on the training data.

The evaluation metrics were:

- Mean RMSE
- Standard deviation of RMSE
- Mean R²

### Baseline Model Results

| Model | Mean CV RMSE | Std CV RMSE | Mean CV R² |
|---|---:|---:|---:|
| Mean Baseline | 77,160.11 | 3,673.25 | -0.0016 |
| Linear Regression | 36,257.55 | 7,443.19 | 0.7717 |
| Ridge (alpha=0.1) | 35,079.40 | 7,278.60 | 0.7873 |
| Log-target Linear Regression | 49,785.87 | 30,338.45 | 0.4661 |

Ridge Regression achieved the best mean CV RMSE among the baseline models.

The log-target Linear Regression model showed considerably higher variation across folds, suggesting less stable cross-validation performance.

## 5. Hyperparameter Tuning

RandomizedSearchCV was used to tune Random Forest and XGBoost.

Both searches used:

- 20 randomly selected parameter combinations.
- 5-fold cross-validation.
- RMSE as the refitting and selection metric.
- R² as an additional evaluation metric.
- Random state of 42.

Each search evaluated 100 model fits.

### Random Forest

Best parameters:

- n_estimators: 300
- max_depth: 20
- min_samples_split: 2
- min_samples_leaf: 2
- max_features: 1.0

Results:

- Mean CV RMSE: 28,623.23
- Standard deviation of RMSE: 5,105.29
- Mean CV R²: 0.8599
- Search training time: 450.34 seconds

### XGBoost

Best parameters:

- n_estimators: 600
- learning_rate: 0.1
- max_depth: 2
- subsample: 1.0
- colsample_bytree: 0.7
- min_child_weight: 3
- reg_lambda: 1

Results:

- Mean CV RMSE: 26,029.57
- Standard deviation of RMSE: 5,275.88
- Mean CV R²: 0.8832
- Search training time: 63.77 seconds

XGBoost achieved better cross-validation performance than Random Forest and completed its search considerably faster.

## 6. Overall Model Comparison

The following table combines cross-validation and final holdout results.

| Model | Mean CV RMSE | Std CV RMSE | Mean CV R² | Holdout RMSE |
|---|---:|---:|---:|---:|
| XGBoost (Tuned) | 26,029.57 | 5,275.88 | 0.8832 | 25,850.94 |
| Random Forest (Tuned) | 28,623.23 | 5,105.29 | 0.8599 | 29,536.46 |
| Ridge (alpha=0.1) | 35,079.40 | 7,278.60 | 0.7873 | 29,200.15 |
| Linear Regression | 36,257.55 | 7,443.19 | 0.7717 | 29,543.93 |
| Log-target Linear Regression | 49,785.87 | 30,338.45 | 0.4661 | 22,749.09 |
| Mean Baseline | 77,160.11 | 3,673.25 | -0.0016 | 87,619.03 |

### Training Time

| Model | Training/Search Time (seconds) |
|---|---:|
| XGBoost (Tuned) | 63.77 |
| Random Forest (Tuned) | 450.34 |
| Ridge (alpha=0.1) | 0.45 |
| Linear Regression | 0.69 |
| Log-target Linear Regression | 0.68 |
| Mean Baseline | 8.40 |

Note: Baseline times represent the recorded cross-validation evaluation time, whereas the Random Forest and XGBoost times include their complete randomized hyperparameter searches. These times are therefore not directly equivalent.

## 7. Final Holdout Test Evaluation

The original 20% test set contained 292 samples.

It was kept separate from cross-validation and hyperparameter tuning and evaluated once after model selection.

### Final Results

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Log-target Linear Regression | 15,035.27 | 22,749.09 | 0.9325 |
| XGBoost (Tuned) | 16,247.42 | 25,850.94 | 0.9129 |
| Ridge (alpha=0.1) | 18,293.84 | 29,200.15 | 0.8888 |
| Random Forest (Tuned) | 17,376.94 | 29,536.46 | 0.8863 |
| Linear Regression | 18,329.98 | 29,543.93 | 0.8862 |
| Mean Baseline | 62,575.93 | 87,619.03 | -0.0009 |

Log-target Linear Regression achieved the lowest holdout RMSE and MAE, as well as the highest holdout R².

XGBoost achieved the second-best holdout RMSE and maintained strong performance across both cross-validation and holdout evaluation.

## 8. Model Selection and Verdict

XGBoost is the recommended model to ship.

The reasons are:

1. It achieved the lowest mean cross-validation RMSE among all evaluated models.
2. It achieved the highest mean cross-validation R².
3. Its holdout RMSE was close to its cross-validation RMSE.
4. It performed better than the tuned Random Forest model.
5. Its tuning process took considerably less time than Random Forest.

However, log-target Linear Regression achieved the best result on the final holdout test set.

Its holdout RMSE was approximately 22,749, compared with approximately 25,851 for XGBoost.

Despite this result, its cross-validation performance was considerably weaker and its standard deviation was much higher. Therefore, the single holdout result is not sufficient evidence that it will consistently outperform XGBoost on other data.

The improvement from feature engineering and ensemble tuning is practically meaningful compared with the mean baseline and the earlier linear models. However, the additional complexity of XGBoost should be considered alongside its improved cross-validation performance.

No formal statistical significance claim is made because repeated or nested cross-validation was not performed.

## 9. Conclusion

This project demonstrated how domain-based feature engineering and ensemble learning can improve regression performance.

Five new features were engineered, and preprocessing was integrated into pipelines to reduce data leakage.

RandomizedSearchCV was used to tune Random Forest and XGBoost over multiple hyperparameters.

XGBoost provided the strongest and most dependable cross-validation performance and is the recommended model for deployment.

Log-target Linear Regression achieved the best result on the single untouched holdout test set, highlighting the importance of comparing cross-validation and final test performance rather than relying on one metric alone.

The project successfully completed feature engineering, model tuning, model comparison, and final holdout evaluation.
