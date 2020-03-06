from pyb4model import pyb4model


def Test_ForSelect():

    # Test Input Types
    assert "sklearn" in str(type(model))
    assert type(max_features) == int
    assert type(min_features) == int
    assert type(cv) == int
    if not isinstance(data_feature, pd.DataFrame):
        raise TypeError("Your data_feature must be a pd.DataFrame object")
    if not isinstance(data_label, pd.DataFrame):
        raise TypeError("Your data_label must be a pd.Series object")
    assert 
    assert problem_type is in ["classification", "regression"]
    assert data_feature.shape[0] == data_label.shape[0]
    
    print("Input Type Test passed")

    return 