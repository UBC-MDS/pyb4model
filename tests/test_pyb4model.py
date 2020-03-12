from pyb4model.pyb4model import fit_and_report, \
                                missing_val, ForSelect, feature_splitter
from pyb4model import pyb4model
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import sklearn.datasets as datasets
import unittest
import pandas as pd
import pytest
import numpy as np


def test_missing_val():

    """
    Test missing_val function
    Check for proper inputs and outputs, throw error otherwise
    """
    df = pd.DataFrame({'A': [1, np.nan, 2, 1, 3], 'B': [np.nan, 4, 5, 6, 3]})
    assert len(missing_val(df, 'delete')
               ) == 3, 'listwise deletion should remove rows\
                                    with missing values'
    assert missing_val(
        df, 'mean').iloc[:, 1][0] == 4.5, 'mean imputation should replace\
                                            missing value with average'
    assert missing_val(
        df, 'knn').iloc[:, 1][0] == 5.5, 'knn imputation should replace missing value\
                                            with nearest neighbour'
    with pytest.raises(ValueError):
        missing_val(pd.DataFrame(), 'delete')
    with pytest.raises(ValueError):
        missing_val(df, 'del')
    with pytest.raises(ValueError):
        missing_val(pd.DataFrame({'A': [1], 'B': [np.nan]}), 'delete')
    with pytest.raises(TypeError):
        missing_val(1, 'delete')


# Here we use knn for regression and classification model and iris dataset
# for testing
class Test_model(unittest.TestCase):
    def test_fit_and_report(self):
        """
        Test function for fit_and_report.
        Check if the return length is correct,\
        if the result is in correct range,if error is raised successfully
        """
        iris = datasets.load_iris(return_X_y=True)
        knn_c = KNeighborsClassifier()
        knn_r = KNeighborsRegressor()
        X = iris[0][1:100]
        y = iris[1][1:100]
        Xv = iris[0][100:]
        yv = iris[1][100:]
        result_r = fit_and_report(knn_r, X, y, Xv, yv, 'regression')
        result_c = fit_and_report(knn_c, X, y, Xv, yv, 'classification')
        # test for output
        self.assertTrue(len(result_r) == 2)
        self.assertTrue(len(result_c) == 2)
        self.assertTrue(0 <= result_r[0] <= 1)
        self.assertTrue(0 <= result_r[1] <= 1)
        self.assertTrue(0 <= result_c[0] <= 1)
        self.assertTrue(0 <= result_c[1] <= 1)
        # test for exception
        self.assertRaises(
            TypeError,
            fit_and_report,
            knn_r,
            X,
            y,
            Xv,
            yv,
            1)
        self.assertRaises(
            TypeError,
            fit_and_report,
            1,
            X,
            y,
            Xv,
            yv,
            'regression')
        self.assertRaises(
            TypeError,
            fit_and_report,
            knn_r,
            1,
            y,
            Xv,
            yv,
            'regression')

#Test for Feature Selection
def test_ForSelect():
    """
    Test function for Feature Selection.
    Checks the return type is a list, not empty and elements\
    in the results are part of the input feature names
    """
    knn_c = KNeighborsClassifier()
    X, y = datasets.load_iris(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    prob = "classification"
    cv = 10

    result = ForSelect(knn_c, X, y, problem_type=prob, cv=cv)

    # Results should be a list with selected features
    assert isinstance(result, list)

    # Results should have at least one element
    assert len(result) >= 1

    # All elements must be included in the input features
    for ele in result:
        assert ele in X.columns

    # Ensure invalid input raises error
    with pytest.raises(ValueError):
        trial = ForSelect(knn_c, X, y, problem_type=3, cv=cv)

    with pytest.raises(TypeError):
        trial = ForSelect(knn_c, X, y, problem_type=prob, cv="3")

    with pytest.raises(TypeError):
        trial = ForSelect("Hello", X, y, problem_type=prob, cv=cv)

    with pytest.raises(TypeError):
        trial = ForSelect(
            knn_c,
            X,
            y,
            max_features="5",
            problem_type=prob,
            cv=cv)

    with pytest.raises(TypeError):
        trial = ForSelect(knn_c, X, pd.DataFrame(y), problem_type=prob, cv=cv)

    with pytest.raises(IndexError):
        trial = ForSelect(knn_c, X, y[:100], problem_type=prob, cv=cv)


def test_feature_splitter():
    """
    Test function for feature splitter
    This function checks if input data if data frame,\
    then splits the data into two parts:
    Which is a tuple containing a list for numeric features and \
    a tupple with numeric features.
    """
    df = {'Name': ['John', 'Micheal', 'Lindsey', 'Adam'],
          'Age': [40, 22, 39, 15],
          'Height(m)': [1.70, 1.82, 1.77, 1.69],
          'Anual Salary(USD)': [40000, 65000, 70000, 15000],
          'Nationality': ['Canada', 'USA', 'Britain', 'Australia'],
          'Marital Status': ['Married', 'Single', 'Maried', 'Single']}
    df = pd.DataFrame(df)

    data_categorical_only = {
        'Name': [
            'John', 'Micheal', 'Lindsey', 'Adam'], 'Nationality': [
            'Canada', 'USA', 'Britain', 'Australia'], 'Marital Status': [
                'Married', 'Single', 'Maried', 'Single']}
    df_cat = pd.DataFrame(data_categorical_only)

    assert feature_splitter(df) == ([
        'Age', 'Height(m)', 'Anual Salary(USD)'], [
        'Name', 'Nationality', 'Marital Status'])
    assert isinstance(feature_splitter(df), tuple)
    assert len(feature_splitter(df)) == 2
    assert feature_splitter(df_cat) == (
        [], ['Name', 'Nationality', 'Marital Status'])
