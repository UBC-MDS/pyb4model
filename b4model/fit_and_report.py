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