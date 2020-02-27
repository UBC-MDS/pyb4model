# pyb4model Python Package

## Summary

## Functions
### Handling Missing Values
- This function will take in a dataframe and handle any missing values by either deleting the row, filling in the value with the average, or filling in the value with the last observation (the user will specify which method to use in the function argument).
- This function will return a dataframe without missing values.

### Split and Scale
- This function will take in a dataframe and split the data into numerical and categorical features.
- This function will return two lists, one list containing the names of the numerical features and one list containing the names of the categorical features.

### Fit and Report
- This function will take in a dataframe, fit a model, and calculate its training and testing scores.
- This function will return the model's training and testing scores.

### Forward Feature Selection
- This function will take in a dataframe, fit a model, and perform forward feature selection.
- This function will return a dataframe with only the selected features.

## Python Ecosystem
