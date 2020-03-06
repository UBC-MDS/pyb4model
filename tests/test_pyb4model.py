from pyb4model import pyb4model
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import sklearn.datasets as datasets
import unittest
import pandas as pd


#Here we use knn for regression and classification model and iris dataset for testing
class Test_model(unittest.TestCase):
    def test_fit_and_report(self):
        iris = datasets.load_iris(return_X_y = True)
        knn_c = KNeighborsClassifier()
        knn_r = KNeighborsRegressor()
        X = iris[0][1:100]
        y =iris[1][1:100]
        Xv = iris[0][100:]
        yv = iris[1][100:]
        result_r = fit_and_report(knn_r, X,y, Xv,yv, 'regression')
        result_c = fit_and_report(knn_c, X,y, Xv,yv, 'classification')
        #test for output
        self.assertTrue(len(result_r)== 2)
        self.assertTrue(len(result_c)== 2)
        self.assertTrue(0 <=result_r[0]<= 1)
        self.assertTrue(0 <=result_r[1]<= 1)
        self.assertTrue(0 <=result_c[0]<= 1)
        self.assertTrue(0 <=result_c[1]<= 1)
        #test for exception 
        self.assertRaises(TypeError, fit_and_report, knn_r, X, y, Xv, yv, 1)
        self.assertRaises(TypeError, fit_and_report, 1, X, y, Xv, yv, 'regression')
        self.assertRaises(TypeError, fit_and_report, knn_r, 1,y, Xv, yv, 'regression')


def test_feature_splitter():
  df = {'Name':['John', 'Micheal', 'Lindsey', 'Adam'],
        'Age':[40, 22, 39, 15],
        'Height(m)':[1.70, 1.82, 1.77, 1.69],
        'Anual Salary(USD)':[40000, 65000, 70000, 15000],
        'Nationality':['Canada', 'USA', 'Britain', 'Australia'],
        'Marital Status':['Married', 'Single', 'Maried', 'Single']} 
   df = pd.DataFrame(df)

  data_categorical_only = {'Name':['John', 'Micheal', 'Lindsey', 'Adam'],
        'Nationality':['Canada', 'USA', 'Britain', 'Australia'],
        'Marital Status':['Married', 'Single', 'Maried', 'Single']} 
  df_cat = pd.DataFrame(data_categorical_only)
        
  assert(feature_splitter(df)!=(['Age', 'Height(m)', 'Anual Salary(USD)'],
    ['Name', 'Nationality', 'Marital Status']))

  assert (type(feature_splitter(df))==tuple
  assert (len(feature_splitter(df))==2
  
  assert feature_splitter(df_cat) == ([], ['Name', 'Nationality', 'Marital Status']), \
    "Dataframes with only categorical data should return a tuple with emply list for numeric\
     and a list for categoric variables"
