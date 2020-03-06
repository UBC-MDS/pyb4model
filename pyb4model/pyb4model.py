import numpy as  np
import pandas as pd
import pytest
from sklearn.model_selection import cross_val_score

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
    model.fit(X, y)
    if m_type.lower().startswith('regress'):
        errors = [mean_squared_error(y, model.predict(X)), mean_squared_error(yv, model.predict(Xv))]
    if m_type.lower().startswith('classif'):
        errors = [1 - model.score(X,y), 1 - model.score(Xv,yv)]        
    return errors



def ForSelect(model,
            data_feature,
            data_label,
            min_features=1,
            max_features=None,
            problem_type='regression'
            cv=3):
        """

        Implementation of forward selection algorithm.
        Search and score with mean cross validation score using feature candidates and
        add features with the best score each step.
        Uses mean squared error for regression, accuracy for classification problem.

        @params
        --------
        model: object            -- sklearn model object
        data_feature: object     -- pandas DataFrame object (features/predictors/explanatory variables)
        data_label: object       -- pandas Series object (labels)
        min_features: integer    -- number of mininum features to select
        max_features: integer    -- number of maximum features to select
        problem_type: string     -- problem type {"classification", "regression"}
        cv: integer              -- k for k-fold-cross-validation

        @returns
        --------
        list                     -- a list of selected column/feature names 


        @example
        --------
        rf = RandomForestClassifier()
        selected_features = ForSelect(rf, 
                                    X_train, 
                                    y_train,
                                    min_features=2, 
                                    max_features=5, 
                                    scoring="neg_mean_square",
                                    problem_type="regression", 
                                    cv=4)
        new_X_train = X_train[selected_features]
        """

        # Test the input & arguments
        Test_ForSelect()

        # Create Empty Feature list
        ftr_ = []

        # Define maximum amount of features
        if max_features is None:
            max_features = X.shape[1]

        # total list of features
        total_ftr = list(range(0, X.shape[1]))

        # define scoring
        if problem_type=="regression":
            scoring='neg_mean_squared_error',
        else:
            scoring='accuracy'
        
        # initialize error score
        best_score = -np.inf

        i = 0

        while len(ftr_) < max_features:
            # remove already selected features
            features_unselected = list(set(total_ftr) - set(ftr_))

            # Initialize potential candidate feature to select
            candidate = None

            # Iterate
            for feature in features_unselected:
                ftr_candidate = ftr_ + [feature]
                eval_score = np.mean(cross_val_score(model,
                                                     X[:, ftr_candidate],
                                                     y,
                                                     cv=cv,
                                                     scoring=scoring))

                # If computed error score is better than our current best score
                if eval_score > best_score:
                    best_score = eval_score  # Overwrite the best_score
                    candidate = feature  # Consider the feature as candidate

            # Add the selected feature
            if candidate is not None:
                ftr_.append(candidate)

                # Report Progress
                i = i + 1
                if i % 5 == 0:
                    print("{} Iterations Done".format(i))
                    print("current best score: {}".format(best_score))

                    print("Current selected features: {}".format(ftr_))
                    print("\n")

            else:
                # End process
                print("{} iterations in total".format(i))
                print("Final selected features: {}".format(ftr_))
                break

        # End Process
        print("{} iterations in total".format(i))
        print("Final selected features: {}".format(ftr_))

        return ftr_

    

        
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
    #code goes here

    pass
