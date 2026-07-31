# Project Overview

This project focuses on predicting the **Fire Weather Index (FWI)** using Algerian forest fire data. The dataset was cleaned and explored through Exploratory Data Analysis (EDA), followed by feature engineering and model training. Multiple regression algorithms, including Linear Regression, Lasso, Ridge, and ElasticNet, were trained and evaluated using **Mean Absolute Error (MAE)** and **R² Score**. Hyperparameter tuning with cross-validation was performed to improve model performance and select the best model.

## Operation Performed

    1. Data Cleaning → 01-dataset-cleaning.ipynb
    2. Exploratory Data Analysis → 02-eda.ipynb`
    3. Feature Engineering → 03-model training.ipynb
    4. Model Training → 03-model training.ipynb
    5. Hyperparametere Tuning → 03-model training.ipynb

> **Note:** Cross Validation for every Regression is Hyperparameter Tuning

## Model Training Workflow

1. Data Cleaning
2. Separate Independent (X) and Dependent (y) Variables
3. Train-Test Split
4. Check Multicollinearity 
5. Feature Scaling / Standardization 
6. Train the Model
7. Evaluate Model Performance

## Models Trained

#### Regression Models
- Linear Regression
- Lasso Regression
- Ridge Regression
- ElasticNet Regression
#### Hyperparameter Tuning

- LassoCV
- RidgeCV
- ElasticNetCV


## General information on Linear Regression Model

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

linreg = LinearRegression()                   # Creates Linreg Model
linreg.fit(X_train_scaled, y_train)           # Using {p1} to train with {p2} target val. .fir is used to map r/s
y_pred = linreg.predict(X_test_scaled)        # {p1} is new feat for model to predict saved in y_pred

mae = mean_absolute_error(y_test, y_pred)     # {p1} is actual val, {p2} is predicted val, mae will calc mae
score = r2_score(y_test, y_pred)              #same, r2 calc variance

print("Mean Square Error", mae)               # The Lower the better, 0 best
print("R2 Score", score)                      # The higher the better, 1 best

plt.scatter(y_test, y_pred)
```

## Interpretation

| Metric                    | Good Value     | Goal     |
| ------------------------- | -------------- | -------- |
| Mean Absolute Error (MAE) | Close to **0** | Minimize |
| R² Score                  | Close to **1** | Maximize |