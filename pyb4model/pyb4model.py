def missing_val(df, method):
    """
    Handles missing values.
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with missing values.
    method: string
        Method to handle missing values.
        'delete', deletes row with missing values
        'avg', replaces missing value with the average
        'last', replaces missing value with the last observation 
    Returns
    -------
    pandas dataframe
        The dataframe without missing values.
    Examples
    --------
    >>> df = pd.DataFrame(np.array([[1, 2, 3], [NaN, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])
    >>> missing_val(df, 'last')
       a  b  c
    0  1  2  3
    1  1  5  6
    2  7  8  9
    """

    # INSERT CODE HERE

from sklearn.metrics import mean_squared_error
ddef fit_and_report(model, X, y, Xv, yv, m_type = 'regression'):
    """
    fits a model and returns the train and validation errors as a list
    
    Arguments
    ---------     
    model -- sklearn classifier model
        The sklearn model
    X -- numpy.ndarray        
        The features of the training set
    y -- numpy.ndarray
        The target of the training set
    Xv -- numpy.ndarray        
        The feature of the validation set
    yv -- numpy.ndarray
        The target of the validation set       
    m_type-- str 
        The type for calculating error (default = 'regression') 
    
    Returns
    -------
    errors -- list
        A list containing train (on X, y) and validation (on Xv, yv) errors
    
    """
    if not isinstance(m_type, str):
        raise TypeError('Input should be a string')
    
    if not "sklearn" in str(type(model)):
        raise TypeError('model should be from sklearn package')
        
    if not "numpy.ndarray" in str(type(X)):
        raise TypeError('Input X should be a numpy array')
    
    if not "numpy.ndarray" in str(type(y)):
        raise TypeError('Input y should be a numpy array')
        
    if not "numpy.ndarray" in str(type(Xv)):
        raise TypeError('Input Xv should be a numpy array')
        
    if not "numpy.ndarray" in str(type(yv)):
        raise TypeError('Input yv should be a numpy array')
        
    model.fit(X, y)
    if m_type.lower().startswith('regress'):
        errors = [mean_squared_error(y, model.predict(X)), mean_squared_error(yv, model.predict(Xv))]
    if m_type.lower().startswith('classif'):
        errors = [1 - model.score(X,y), 1 - model.score(Xv,yv)]        
    return errors




import numpy as  np
from sklearn.model_selection import cross_val_score

class ForSelect:
    def __init__(self, model,
                 min_features=None,
                 max_features=None,
                 scoring=None,
                 cv=None):
        """
        Defining Class
        @params
        --------
        model: object            -- sklearn model object
        min_features: integer    -- number of mininum features to select
        max_features: integer    -- number of maximum features to select
        scoring:  string         -- sklearn scoring metric
        cv: integer              -- k for k-fold-cross-validation
        @example
        --------
        fs = ForSelect
        """

    def fit(self, X, y):
        """
        - Implementation of forward selection algorithm.
        - Search and score with mean cross validation score using feature candidates and
          add features with the best score each step.
        - Return dataset with selected features.
        @params
        --------
        X: array       -- training dataset (features)
        y: array       -- training dataset (labels)
        @returns
        --------
        self          -- with updated self.ftr_
        @example
        -------
        rf = RandomForestClassifier()
        fs = ForSelect(rf, min_features=2, max_features=5)
        fs.fit(X_train, y_train)
        fs.ftr_
        """

        
import pandas as pd

def feature_splitter(x):
    """ Splits dataset column names into a tuple of categorical and numerical lists
    Parameters
    ----------
    x : DateFrame
    Returns
    -------
    tuple: 
        tuple of two lists
    
    Example
    -------
    >>> feature_splitter(data)
    ([categorical:],[numerical: ])
    """
    # Identify the categorical and numeric columns
    assert data.shape[1]>1 and data.shape[0]>1, "Your data file in not valid, dataframe should have at least\
                                                one column and one row"    
    if not isinstance(data, pd.DataFrame):
        raise Exception('the input data should be a data frame')
        
    d_types = data.dtypes
    categorical = []
    numerical = []
    
    for data_type, features in zip(d_types, d_types.index):
        if data_type == "object":
            categorical.append(features)
        else:
            numerical.append(features)
            
    assert len(numerical) + len(categorical)==data.shape[1], "categorical and numerical variable list must match\
                                                                number of columns in the data frame"
    return numerical, categorical
