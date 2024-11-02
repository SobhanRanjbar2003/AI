from typing import List, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class MultiNodeCategoricalDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, criterion="entropy"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiNodeCategoricalDecisionTree":
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)
        self.feature_importances_ = self._calculate_feature_importances()
        return self

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> Dict[str, Any]:
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or n_classes == 1
        ):
            return {
                "type": "leaf",
                "value": np.argmax(np.bincount(y)),
                "n_samples": n_samples,
            }

        best_split = self._best_split(X, y)

        if best_split is None:
            return {
                "type": "leaf",
                "value": np.argmax(np.bincount(y)),
                "n_samples": n_samples,
            }

        node = {
            "type": "branch",
            "feature": best_split["feature"],
            "children": {},
            "n_samples": n_samples,
        }

        feature_values = np.unique(X[:, best_split["feature"]])
        for value in feature_values:
            mask = X[:, best_split["feature"]] == value
            if np.any(mask):
                node["children"][value] = self._build_tree(X[mask], y[mask], depth + 1)

        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_split = None

        current_impurity = (
            self._calculate_entropy(y)
            if self.criterion == "entropy"
            else self._calculate_gini(y)
        )

        for feature_idx in range(n_features):
            feature_values = np.unique(X[:, feature_idx])

            # Skip if feature has only one value
            if len(feature_values) < 2:
                continue

            feature_gain = current_impurity

            weighted_child_impurity = 0
            for value in feature_values:
                mask = X[:, feature_idx] == value
                prop = np.sum(mask) / n_samples

                if prop > 0:
                    child_impurity = (
                        self._calculate_entropy(y[mask])
                        if self.criterion == "entropy"
                        else self._calculate_gini(y[mask])
                    )
                    weighted_child_impurity += prop * child_impurity

            gain = feature_gain - weighted_child_impurity

            if gain > best_gain:
                best_gain = gain
                best_split = {"feature": feature_idx, "gain": gain}

        return best_split

    def _calculate_feature_importances(self) -> np.ndarray:
        importances = np.zeros(self.n_features_)

        def _compute_importance(node, total_samples):
            if node["type"] == "branch":
                feature = node["feature"]
                n_samples = node["n_samples"]

                importance = n_samples / total_samples
                importances[feature] += importance

                for child in node["children"].values():
                    _compute_importance(child, total_samples)

        _compute_importance(self.tree_, self.tree_["n_samples"])

        importances = (
            importances / np.sum(importances)
            if np.sum(importances) > 0
            else importances
        )
        return importances

    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calculate entropy using the formula: -sum(p * log2(p))"""
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
        return entropy

    def _calculate_gini(self, y: np.ndarray) -> float:
        """Calculate Gini index using the formula: 1 - sum(p^2)"""
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        gini = 1 - np.sum(np.square(probabilities))
        return gini

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        node = self.tree_

        while node["type"] == "branch":
            feature_index = node["feature"]
            feature_value = x[feature_index]

            if feature_value not in node["children"]:
                leaf_values = []
                for child in node["children"].values():
                    if child["type"] == "leaf":
                        leaf_values.extend([child["value"]] * child["n_samples"])
                    elif "value" in child:
                        leaf_values.append(child["value"])

                if leaf_values:
                    return np.bincount(leaf_values).argmax()
                else:
                    return 0

            node = node["children"][feature_value]

        if node["type"] == "leaf":
            return node["value"]

        return 0
