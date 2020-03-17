![Release](https://github.com/UBC-MDS/pyb4model/workflows/Release/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/UBC-MDS/pyb4model/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/pyb4model)
[![Documentation Status](https://readthedocs.org/projects/pyb4model/badge/?version=latest)](https://pyb4model.readthedocs.io/en/latest/?badge=latest)

# pyb4model Python Package

## Summary
This project aims to build a Python package that elegantly performs data pre-processing in a fast and easy manner. With four separate functions that will come along with the pyb4model package, users will have greater flexibility in handling many different types of datasets in the wild or those collected by them. With the pyb4model package, users will be able to smoothly pre-process their data and have it ready for the machine learning model of their choice.

## Functions
`missing_val`
- This function will take in a dataframe and handle any missing values by either deleting the row, filling in the value with the average, or filling in the value with the last observation (the user will specify which method to use in the function argument).
- This function will return a dataframe without missing values.

`feature_splitter`
- This function will take in a dataframe and split the data into numerical and categorical features.
- This function will return two lists, one list containing the names of the numerical features and one list containing the names of the categorical features.

`fit_and_report`
- This function will take in data, fit a model, and calculate its training and validation scores.
- This function will return the model's training and validation scores.

`ForSelect`
- This function will take in data, fit a model, and perform forward feature selection.
- This function will return a dataframe with only the selected features.

### Installation

```
pip install -i https://test.pypi.org/simple/ pyb4model
```

### Usage

```python3
>>> from pyb4model import pyb4model as pbm
>>> from sklearn.metrics import mean_squared_error
>>> from sklearn.impute import KNNImputer
>>> from sklearn.model_selection import cross_val_score
>>> import numpy as np
>>> import pandas as pd
```
##### Missing Value Function

```python3
    Example
    --------
    df = pd.DataFrame(np.array([[1, 2, 3], [NaN, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
    pbm.missing_val(df, 'last')
```

##### Fit and Report Function
```python3
    Example
    --------
    iris = datasets.load_iris(return_X_y = True)
    knn_c = KNeighborsClassifier()
    knn_r = KNeighborsRegressor()
    X = iris[0][1:100]
    y =iris[1][1:100]
    Xv = iris[0][100:]
    yv = iris[1][100:]
    result_r = pbm.fit_and_report(knn_r, X,y, Xv,yv, 'regression')
```
##### Forward Selection Function
```python3
    Example
    --------
    rf = RandomForestClassifier()
    selected_features = pbm.ForSelect(rf,
                                X_train,
                                y_train,
                                min_features=2,
                                max_features=5,
                                scoring="neg_mean_square",
                                problem_type="regression",
                                cv=4)
    new_X_train = X_train[selected_features]
```
##### Feature Splitter Function
```python3
    Example
    -------
    df = {'Name': ['John', 'Micheal', 'Lindsey', 'Adam'],
          'Age': [40, 22, 39, 15],
          'Height(m)': [1.70, 1.82, 1.77, 1.69],
          'Anual Salary(USD)': [40000, 65000, 70000, 15000],
          'Nationality': ['Canada', 'USA', 'Britain', 'Australia'],
          'Marital Status': ['Married', 'Single', 'Maried', 'Single']}
    df = pd.DataFrame(df)
    pbm.feature_splitter(data)
```
## Dependencies
|Package|Version|
|-------|-------|
|[python](https://www.python.org/downloads/release/python-370/) |^3.7   |
|[pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) |^1.0   |
|[sklearn](https://scikit-learn.org/stable/install.html)|^0.22.1|
|[numpy](https://scipy.org/install.html)  |^1.18.1|
|[pytest](https://docs.pytest.org/en/latest/getting-started.html) |^5.3.5 |

## Documentation

The complete documentation of this package is available [here](https://pyb4model.readthedocs.io/en/latest/?badge=latest)
## Python Ecosystem

The Python package `sklearn` provides extensive classes of Machine Learning models and functions for feature selection and engineering. Some of the feature selection modules that `sklearn` has are: recursive feature elimination, univariate feature selection, and L1-based feature elimination. However, it does not have Forward Feature Selection.

Furthermore, it is a tedious job to write numerous lines of code to clean data, split, scale, fit and report scores for baseline models or models with default settings, once you are used to `sklearn`. 

In this sense, our package can save programmer's time by providing a wrapper of `sklearn` and using them with a few lines of code instead of copy and pasting a long series of code.
