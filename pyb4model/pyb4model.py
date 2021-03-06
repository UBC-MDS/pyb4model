from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


def missing_val(df, method):
    """
    Handles missing values.
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with missing values.
    method : string
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
    >>> df = pd.DataFrame(np.array([[1, 2, 3], [np.NaN, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])
    >>> missing_val(df, 'knn')
       a  b  c
    0  1  2  3
    1  1  5  6
    2  7  8  9
    """

    # tests

    if method not in ['delete', 'mean', 'knn']:
        raise ValueError(
            'valid methods only include "delete", "mean", and "knn"')

    if not isinstance(
            df,
            pd.DataFrame) and not isinstance(
            df,
            np.ndarray) and not isinstance(
                df,
            pd.Series):
        raise TypeError('df must be a dataframe, series, or array')

    if df.empty:  # edge case
        raise ValueError('dataframe cannot be empty')

    if all(df.dtypes != np.number):  # edge case
        raise ValueError('dataframe must have at least one numerical column')

    for i in range(len(df.columns)):  # edge case
        if df.iloc[:, i].isnull().sum() == len(df):
            raise ValueError('dataframe cannot columns with all NaN values')

    # function

    if method == 'delete':
        df = df.dropna()

    if method == 'mean':
        df_num = df.select_dtypes(include=np.number)
        df_cat = df.select_dtypes(exclude=np.number)
        df_num = df_num.fillna(df.mean())
        df = pd.concat([df_num, df_cat], axis=1)

    if method == 'knn':
        df_num = df.select_dtypes(include=np.number)
        df_cat = df.select_dtypes(exclude=np.number)
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        df_num = pd.DataFrame(imputer.fit_transform(df_num))
        df = pd.concat([df_num, df_cat], axis=1)

    return df


def fit_and_report(model, X, y, Xv, yv, m_type='regression'):
    """
    Fits a model and returns the train and validation errors as a list

    Parameters
    ---------
    model : sklearn classifier model
        The sklearn model
    X : numpy.ndarray
        The features of the training set
    y : numpy.ndarray
        The target of the training set
    Xv : numpy.ndarray
        The feature of the validation set
    yv : numpy.ndarray
        The target of the validation set
    m_type : str
        The type for calculating error (default = 'regression')


    Returns
    -------
    errors : list
        A list containing train (on X, y) and validation (on Xv, yv) errors

    Examples
    --------
    >>> iris = datasets.load_iris(return_X_y = True)
    >>> knn_c = KNeighborsClassifier()
    >>> knn_r = KNeighborsRegressor()
    >>> X = iris[0][1:100]
    >>> y =iris[1][1:100]
    >>> Xv = iris[0][100:]
    >>> yv = iris[1][100:]
    >>> fit_and_report(knn_r, X,y, Xv,yv, 'regression')
    [0.0, 1.0]
    """
    if not isinstance(m_type, str):
        raise TypeError('Input should be a string')

    if "sklearn" not in str(type(model)):
        raise TypeError('model should be from sklearn package')

    if "numpy.ndarray" not in str(type(X)):
        raise TypeError('Input X should be a numpy array')

    if "numpy.ndarray" not in str(type(y)):
        raise TypeError('Input y should be a numpy array')

    if "numpy.ndarray" not in str(type(Xv)):
        raise TypeError('Input Xv should be a numpy array')

    if "numpy.ndarray" not in str(type(yv)):
        raise TypeError('Input yv should be a numpy array')

    model.fit(X, y)
    if m_type.lower().startswith('regress'):
        errors = [
            mean_squared_error(
                y, model.predict(X)), mean_squared_error(
                yv, model.predict(Xv))]
    if m_type.lower().startswith('classif'):
        errors = [1 - model.score(X, y), 1 - model.score(Xv, yv)]
    return errors


def ForSelect(
        model,
        data_feature,
        data_label,
        max_features=None,
        problem_type='regression',
        cv=3):
    """
    Implementation of forward selection algorithm.
    Search and score with mean cross validation score
    using feature candidates and
    add features with the best score each step.
    Uses mean squared error for regression,
    accuracy for classification problem.

    Parameters
    --------
    model : object
        sklearn model object
    data_feature : object
        pandas DataFrame object (features/predictors)
    data_label : object
        pandas Series object (labels)
    max_features : integer
        number of maximum features to select
    problem_type : string
        problem type {"classification", "regression"}
    cv : integer
        k for k-fold-cross-validation

    Returns
    --------
    list
        a list of selected column/feature names


    Example
    --------
    >>> rf = RandomForestClassifier()
    >>> iris = datasets.load_iris(return_X_y = True)
    >>> X_train = pd.DataFrame(iris[0][1:100])
    >>> y_train = pd.Series(iris[1][1:100])
    >>> ForSelect(rf,
                    data_feature=X_train,
                    data_label=y_train,
                    max_features=5,
                    problem_type="classification",
                    cv=2)
    """

    # Test Input Types
    if "sklearn" not in str(type(model)):
        raise TypeError("Your Model should be sklearn model")

    if (not isinstance(max_features, int)) and (max_features is not None):
        raise TypeError("Your max number of features should be an integer")

    if not isinstance(cv, int):
        raise TypeError("Your cross validation number should be an integer")

    if not isinstance(data_feature, pd.DataFrame):
        raise TypeError("Your data_feature must be a pd.DataFrame object")

    if not isinstance(data_label, pd.Series):
        raise TypeError("Your data_label must be a pd.Series object")

    if problem_type not in ["classification", "regression"]:
        raise ValueError(
            "Your problem should be 'classification' or 'regression'")

    if data_feature.shape[0] != data_label.shape[0]:
        raise IndexError(
            "Number of rows are different in training feature and label")

    # Create Empty Feature list
    ftr_ = []

    # Define maximum amount of features
    if max_features is None:
        max_features = data_feature.shape[1]

    # total list of features
    total_ftr = list(range(0, data_feature.shape[1]))

    # define scoring
    if problem_type == "regression":
        scoring = 'neg_mean_squared_error'
    else:
        scoring = 'accuracy'

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
            eval_score = np.mean(
                cross_val_score(
                    model,
                    data_feature[ftr_candidate],
                    data_label,
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

        else:
            # End process
            break

    # End Process

    print("Final selected features: {}".format(ftr_))

    return ftr_


def feature_splitter(data):
    """
    Splits dataset column names into a tuple of categorical and numerical lists


    Parameters
    ----------
    x : DateFrame


    Returns
    -------
    tuple
        tuple of two lists


    Example
    -------
    >>> df = {'Name': ['John', 'Micheal', 'Lindsey', 'Adam'],
              'Age': [40, 22, 39, 15],
              'Height(m)': [1.70, 1.82, 1.77, 1.69],
              'Anual Salary(USD)': [40000, 65000, 70000, 15000],
              'Nationality': ['Canada', 'USA', 'Britain', 'Australia'],
              'Marital Status': ['Married', 'Single', 'Maried', 'Single']}
    >>> df = pd.DataFrame(df)
    >>> feature_splitter(df)
    (['Age', 'Height(m)', 'Anual Salary(USD)'],
    ['Name', 'Nationality', 'Marital Status'])
    """
    # Identify the categorical and numeric columns
    assert data.shape[1] > 1 and data.shape[0] > 1, "Your data file in not valid, dataframe should have at least\
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

    assert len(numerical) + \
        len(categorical) == data.shape[1], "categorical and numerical variable list must match\
                                                                df shape"
    numerical = pd.DataFrame(numerical, columns=['Numerical'])
    categorical = pd.DataFrame(categorical, columns=['Categorical'])
    result = pd.concat([numerical, categorical], axis=1, ignore_index=False)
    result = result.replace(np.nan, '-')
    return result
