import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


class DecisionStumpClassifier(BaseEstimator, ClassifierMixin):
    """Generic Attribute Threshold Binary Classifier

    Attributes
    ----------
    attribute_index : int
        The attribute to consider for the classification.
    threshold : float
        The threshold value for classification rule.
    direction : int, optional
        A multiplicative constant (1 or -1) to choose the "direction" of the stump. Defaults to 1. If -1, the stump
        will predict the "negative" class (generally -1 or 0), and if 1, the stump will predict the second class (generally 1).

    """

    def __init__(self, attribute_index, threshold, direction=1):
        super(DecisionStumpClassifier, self).__init__()
        self.attribute_index = attribute_index
        self.threshold = threshold
        self.direction = direction

    def fit(self, X, y):
        # Only verify that we are in the binary classification setting, with support for transductive learning.
        if isinstance(y, np.ma.MaskedArray):
            self.classes_ = np.unique(y[np.logical_not(y.mask)])
        else:
            self.classes_ = np.unique(y)

        # This label encoder is there for the predict function to be able to return any two classes that were used
        # when fitting, for example {-1, 1} or {0, 1}.
        self.le_ = LabelEncoder()
        self.le_.fit(self.classes_)
        self.classes_ = self.le_.classes_

        if not len(self.classes_) == 2:
            raise ValueError(
                'DecisionStumpsVoter only supports binary classification')
        # assert len(self.classes_) == 2, "DecisionStumpsVoter only supports binary classification"
        return self

    def predict(self, X):
        """Returns the output of the classifier, on a sample X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        predictions : array-like, shape = [n_samples]
            Predicted class labels.

        """
        check_is_fitted(self, 'classes_')
        return self.le_.inverse_transform(
            np.argmax(self.predict_proba(X), axis=1))

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """
        check_is_fitted(self, 'classes_')
        X = np.asarray(X)
        probas = np.zeros((X.shape[0], 2))
        positive_class = np.argwhere(
            X[:, self.attribute_index] > self.threshold)
        negative_class = np.setdiff1d(range(X.shape[0]), positive_class)
        probas[positive_class, 1] = 1.0
        probas[negative_class, 0] = 1.0

        if self.direction == -1:
            probas = 1 - probas

        return probas

    def predict_proba_t(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """

        X = np.ones(X.shape)
        check_is_fitted(self, 'classes_')
        X = np.asarray(X)
        probas = np.zeros((X.shape[0], 2))
        positive_class = np.argwhere(
            X[:, self.attribute_index] > self.threshold)
        negative_class = np.setdiff1d(range(X.shape[0]), positive_class)
        probas[positive_class, 1] = 1.0
        probas[negative_class, 0] = 1.0

        if self.direction == -1:
            probas = 1 - probas

        return probas

    def reverse_decision(self):
        self.direction *= -1

class ClassifiersGenerator(BaseEstimator, TransformerMixin):
    """Base class to create a set of voters using training samples, and then transform a set of examples in
    the voters' output space.

    Attributes
    ----------
    self_complemented : bool, optional
        Whether or not a binary complement voter must be generated for each voter. Defaults to False.
    voters : ndarray of voter functions
        Once fit, contains the voter functions.

    """

    def __init__(self, self_complemented=False):
        super(ClassifiersGenerator, self).__init__()
        self.self_complemented = self_complemented

    def fit(self, X, y=None):
        """Generates the voters using training samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data on which to base the voters.
        y : ndarray of shape (n_labeled_samples,), optional
            Input labels, usually determines the decision polarity of each voter.

        Returns
        -------
        self

        """
        raise NotImplementedError

    def transform(self, X):
        """Transforms the input points in a matrix of classification, using previously learned voters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to classify.

        Returns
        -------
        ndarray of shape (n_samples, n_voters)
            The voters' decision on each example.

        """
        check_is_fitted(self, 'estimators_')
        return np.array([voter.predict(X) for voter in self.estimators_]).T

class TreeClassifiersGenerator(ClassifiersGenerator):

    def __init__(self, random_state=42, max_depth=2, self_complemented=True,
                 criterion="gini", splitter="best", n_trees=100,
                 distribution_type="uniform", low=0, high=10,
                 attributes_ratio=0.6, examples_ratio=0.95):
        super(TreeClassifiersGenerator, self).__init__(self_complemented)
        self.max_depth = max_depth
        self.criterion = criterion
        self.splitter = splitter
        self.n_trees = n_trees
        if type(random_state) is int:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
        self.distribution_type = distribution_type
        self.low = low
        self.high = high
        self.attributes_ratio = attributes_ratio
        self.examples_ratio = examples_ratio

    def fit(self, X, y=None):
        estimators_ = []
        self.attribute_indices = np.array(
            [self.sub_sample_attributes(X) for _ in range(self.n_trees)])
        self.example_indices = np.array(
            [self.sub_sample_examples(X) for _ in range(self.n_trees)])
        for i in range(self.n_trees):
            estimators_.append(DecisionTreeClassifier(criterion=self.criterion,
                                                      splitter=self.splitter,
                                                      max_depth=self.max_depth).fit(
                X[:, self.attribute_indices[i, :]][self.example_indices[i], :],
                y[self.example_indices[i, :]]))
        self.estimators_ = np.asarray(estimators_)
        return self

    def sub_sample_attributes(self, X):
        n_attributes = X.shape[1]
        attributes_indices = np.arange(n_attributes)
        kept_indices = self.random_state.choice(attributes_indices, size=int(
            self.attributes_ratio * n_attributes), replace=True)
        return kept_indices

    def sub_sample_examples(self, X):
        n_examples = X.shape[0]
        examples_indices = np.arange(n_examples)
        kept_indices = self.random_state.choice(examples_indices, size=int(
            self.examples_ratio * n_examples), replace=True)
        return kept_indices

    def choose(self, chosen_columns):
        self.estimators_ = self.estimators_[chosen_columns]
        self.attribute_indices = self.attribute_indices[chosen_columns, :]
        self.example_indices = self.example_indices[chosen_columns, :]


class StumpsClassifiersGenerator(ClassifiersGenerator):
    """Decision Stump Voters transformer.

    Parameters
    ----------
    n_stumps_per_attribute : int, optional
        Determines how many decision stumps will be created for each attribute. Defaults to 10.
        No stumps will be created for attributes with only one possible value.
    self_complemented : bool, optional
        Whether or not a binary complement voter must be generated for each voter. Defaults to False.

    """

    def __init__(self, n_stumps_per_attribute=10, self_complemented=False,
                 check_diff=False):
        super(StumpsClassifiersGenerator, self).__init__(self_complemented)
        self.n_stumps_per_attribute = n_stumps_per_attribute
        self.check_diff = check_diff

    def fit(self, X, y=None):
        """Fits Decision Stump voters on a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data on which to base the voters.
        y : ndarray of shape (n_labeled_samples,), optional
            Only used to ensure that we are in the binary classification setting.

        Returns
        -------
        self

        """
        minimums = np.min(X, axis=0)
        maximums = np.max(X, axis=0)
        if y.ndim > 1:
            y = np.reshape(y, (y.shape[0],))
        ranges = (maximums - minimums) / (self.n_stumps_per_attribute + 1)
        if self.check_diff:
            nb_differents = [np.unique(col) for col in np.transpose(X)]
            self.estimators_ = []
            for i in range(X.shape[1]):
                nb_different = nb_differents[i].shape[0]
                different = nb_differents[i]
                if nb_different - 1 < self.n_stumps_per_attribute:
                    self.estimators_ += [DecisionStumpClassifier(i,
                                                                 (different[
                                                                      stump_number] +
                                                                  different[
                                                                      stump_number + 1]) / 2,
                                                                 1).fit(X, y)
                                         for stump_number in
                                         range(int(nb_different) - 1)]
                    if self.self_complemented:
                        self.estimators_ += [DecisionStumpClassifier(i,
                                                                     (different[
                                                                          stump_number] +
                                                                      different[
                                                                          stump_number + 1]) / 2,
                                                                     -1).fit(X,
                                                                             y)
                                             for stump_number in
                                             range(int(nb_different) - 1)]
                else:
                    self.estimators_ += [DecisionStumpClassifier(i,
                                                                 minimums[i] +
                                                                 ranges[
                                                                     i] * stump_number,
                                                                 1).fit(X, y)
                                         for stump_number in range(1,
                                                                   self.n_stumps_per_attribute + 1)
                                         if ranges[i] != 0]

                    if self.self_complemented:
                        self.estimators_ += [DecisionStumpClassifier(i,
                                                                     minimums[
                                                                         i] +
                                                                     ranges[
                                                                         i] * stump_number,
                                                                     -1).fit(X,
                                                                             y)
                                             for stump_number in range(1,
                                                                       self.n_stumps_per_attribute + 1)
                                             if ranges[i] != 0]
        else:
            self.estimators_ = [DecisionStumpClassifier(i, minimums[i] + ranges[
                i] * stump_number, 1).fit(X, y)
                                for i in range(X.shape[1]) for stump_number in
                                range(1, self.n_stumps_per_attribute + 1)
                                if ranges[i] != 0]

            if self.self_complemented:
                self.estimators_ += [DecisionStumpClassifier(i, minimums[i] +
                                                             ranges[
                                                                 i] * stump_number,
                                                             -1).fit(X, y)
                                     for i in range(X.shape[1]) for stump_number
                                     in
                                     range(1, self.n_stumps_per_attribute + 1)
                                     if ranges[i] != 0]
        self.estimators_ = np.asarray(self.estimators_)
        return self

    def choose(self, chosen_columns):
        self.estimators_ = self.estimators_[chosen_columns]
