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

