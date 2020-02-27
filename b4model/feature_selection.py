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
        self.max_features = max_features

        if min_features is None:
            self.min_features = 1
        else:
            self.min_features = min_features

        self.model = model
        self.scoring = scoring
        self.cv = cv
        self.ftr_ = []
        return



    def fit(self, X, y):
        """
        Implementation of forward selection algorithm.
        Search and score with mean cross validation score using feature candidates and
        add features with the best score each step.

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

        if self.max_features is None:
            self.max_features = X.shape[1]

        # total list of features
        total_ftr = list(range(0, X.shape[1]))

        # initialize error score

        best_score = -np.inf

        i = 0
        while len(self.ftr_) < self.max_features:
            # remove already selected features
            features_unselected = list(set(total_ftr) - set(self.ftr_))

            # Initialize potential candidate feature to select
            candidate = None
            # Iterate
            for feature in features_unselected:
                ftr_candidate = self.ftr_ + [feature]
                eval_score = np.mean(cross_val_score(self.model,
                                                     X[:, ftr_candidate],
                                                     y,
                                                     cv=self.cv,
                                                     scoring=self.scoring))

                # If computed error score is better than our current best score
                if eval_score > best_score:
                    best_score = eval_score  # Overwrite the best_score
                    candidate = feature  # Consider the feature as candidate

            # Add the selected feature
            if candidate is not None:
                self.ftr_.append(candidate)

                # Report Progress
                i = i + 1
                if i % 5 == 0:
                    print("{} Iterations Done".format(i))
                    print("current best score: {}".format(best_score))

                    print("Current selected features: {}".format(self.ftr_))
                    print("\n")

            else:
                # End process
                print("{} iterations in total".format(i))
                print("Final selected features: {}".format(self.ftr_))
                break

        # End Process
        print("{} iterations in total".format(i))
        print("Final selected features: {}".format(self.ftr_))

        return
