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
