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

## Python Ecosystem

The Python package `sklearn` provides extensive classes of Machine Learning models and functions for feature selection and engineering. Some of the feature selection modules that `sklearn` has are: recursive feature elimination, univariate feature selection, and L1-based feature elimination. However, it does not have Forward Feature Selection.

Furthermore, it is a tedious job to write numerous lines of code to clean data, split, scale, fit and report scores for baseline models or models with default settings, once you are used to `sklearn`. 

In this sense, our package can save programmer's time by providing a wrapper of `sklearn` and using them with a few lines of code instead of copy and pasting a long series of code.
