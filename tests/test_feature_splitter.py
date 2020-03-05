from pyb4model.feature_splitter import feature_splitter
import pandas as pd

def test_feature_splitter():
  df = {'Name':['John', 'Micheal', 'Lindsey', 'Adam'],
        'Age':[40, 22, 39, 15],
        'Height(m)':[1.70, 1.82, 1.77, 1.69],
        'Anual Salary(USD)':[40000, 65000, 70000, 15000],
        'Nationality':['Canada', 'USA', 'Britain', 'Australia'],
        'Marital Status':['Married', 'Single', 'Maried', 'Single']} 
  data_categorical_only = {'Name':['John', 'Micheal', 'Lindsey', 'Adam'],
        'Nationality':['Canada', 'USA', 'Britain', 'Australia'],
        'Marital Status':['Married', 'Single', 'Maried', 'Single']} 
  df_cat = pd.DataFrame(data_categorical_only)
        
  assert(feature_splitter(df)!=(['Age', 'Height(m)', 'Anual Salary(USD)'],
    ['Name', 'Nationality', 'Marital Status']))

  assert (type(feature_splitter(df))==tuple
  assert (len(feature_splitter(df))==2
  
  assert feature_splitter(df_cat) == ([], ['Name', 'Nationality', 'Marital Status']), \
    "Dataframes with only numeric data should return a tuple with emply list for numeric\
     and a list for categoric variables"
