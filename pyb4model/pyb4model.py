from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
import numpy as  np
import pandas as pd

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
        'mean', replaces missing values with the averages
        'knn', replaces missing values with nearest neighbour
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
    
    # tests
    
    if method not in ['delete', 'mean', 'knn']:
        raise ValueError('valid methods only include "delete", "mean", and "regression"')
    
    if not isinstance(df, pd.DataFrame) and not isinstance(df, np.ndarray) and not isinstance(df, pd.Series):
        raise TypeError('df must be a dataframe, series, or array')
        
    if df.empty: # edge case
        raise ValueError('dataframe cannot be empty')
        
    for i in range(len(df.columns)): # edge case
        if df.iloc[:,i].isnull().sum()==len(df):
            raise ValueError('dataframe cannot columns with all NaN values')
    
    # function
    
    if method=='delete':
        df = df.dropna()
    
    if method=='mean':
        df = df.fillna(df.mean())
        
    if method=='knn':
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        df = pd.DataFrame(imputer.fit_transform(df))
    
    return df


def fit_and_report(model, X, y, Xv, yv, m_type = 'regression'):
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
    #code goes here

    pass