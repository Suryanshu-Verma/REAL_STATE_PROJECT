Here's a formatted README file based on the provided code:

---

# Project Name: Regression Analysis - Predicting House Prices in Boston City

## Overview:

This project aims to predict house prices in Boston City using regression analysis. The dataset used for this analysis contains various features such as crime rate, zoning information, property tax rates, and more. Different regression models are trained and evaluated to determine the most accurate prediction.

## Table of Contents:

1. [Importing Libraries](#importing-libraries)
2. [Importing Datasets](#importing-datasets)
3. [Generating Profile Using Pandas Profiling](#generating-profile-using-pandas-profiling)
4. [Visualizing the Dataset](#visualizing-the-dataset)
5. [Independent Features & Dependent Label](#independent-features--dependent-label)
6. [Validating Missing Values](#validating-the-missing-values-in-the-dataset)
7. [Train and Test Split](#train-and-test-split)
8. [Model Selection](#model-selection)
    - Linear Regression
    - Decision Tree Regression
    - Random Forest Regression
9. [Grid Search CV](#grid-search-cv)
10. [Using the Model](#using-the-model)
11. [Conclusion](#conclusion)

## Importing Libraries <a name="importing-libraries"></a>

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from ydata_profiling import ProfileReport
```

## Importing Datasets <a name="importing-datasets"></a>

```python
# Reading The Data By Pandas
data = pd.read_csv("data.csv")
# Creating 'X' Matrix For Independent Features In The Dataset
X = data.iloc[:,:-1].values
# Creating 'Y' Matrix For Dependent Features In The Dataset
Y = data.iloc[:,-1].values
```

## Generating Profile Using Pandas Profiling <a name="generating-profile-using-pandas-profiling"></a>

```python
profile = ProfileReport(data)
profile.to_notebook_iframe()
```

## Visualizing the Dataset <a name="visualizing-the-dataset"></a>

```python
# For plotting histogram
data.hist(bins=50, figsize=(20, 15))
```

## Independent Features & Dependent Label <a name="independent-features--dependent-label"></a>

```python
print("Matrix For Independent Features In The Dataset:\n", X)
print("Matrix For Dependent Features In The Dataset:\n", Y.reshape(len(Y),1))
```

## Validating Missing Values <a name="validating-the-missing-values-in-the-dataset"></a>

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, :14] = imputer.fit_transform(X[:, :14])
```

## Train and Test Split <a name="train-and-test-split"></a>

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```

## Model Selection <a name="model-selection"></a>

Three regression models were evaluated: Linear Regression, Decision Tree Regression, and Random Forest Regression.

## Grid Search CV <a name="grid-search-cv"></a>

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid
parameters = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'criterion': ["squared_error", "absolute_error", "friedman_mse", "poisson"], 
    'max_depth': [20]
}

# Initialize a RandomForestRegressor
model = RandomForestRegressor()

# Initialize GridSearchCV with the model and parameter grid
GCV = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
GCV.fit(X_train, Y_train)
```

## Using the Model <a name="using-the-model"></a>

```python
from joblib import dump, load
GCV = load('REAL.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
prediction = GCV.predict(features)
print(f"The predicted price of the house is: {int(prediction)*1000} $ in Boston City")
```

## Conclusion <a name="conclusion"></a>

The project successfully predicts house prices in Boston City using various regression techniques. The best model achieved an R-squared score of X.XX, indicating [insert conclusion here].

---

This README provides an overview of the project, including data preprocessing, model selection, evaluation, and usage. It serves as a guide for users interested in understanding and replicating the analysis. Adjustments can be made to the

 markdown formatting and content as needed.
