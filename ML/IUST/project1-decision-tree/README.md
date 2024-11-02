# Multi-Node Categorical Decision Tree Classifier

## Introduction

This project implements a multi-node categorical decision tree classifier that is compatible with scikit-learn. Unlike binary decision trees, this classifier is designed to work with categorical features and can have multiple branches at each node. The implementation is based on the `MultiNodeCategoricalDecisionTree` class, which inherits from scikit-learn's `BaseEstimator` and `ClassifierMixin`.

The purpose of this project is to provide a hands-on experience in implementing a decision tree algorithm from scratch while ensuring compatibility with a popular machine learning library.

## Setting Up the Environment

To set up the project environment and install the necessary libraries, follow these steps:

1. Ensure you have Python 3.7 or higher installed on your system.

2. Open a terminal and navigate to the project directory.

3. Create a virtual environment:
   ```
   python -m venv venv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

5. Install the required libraries:
   ```
   pip install numpy -r requirements.txt
   ```

Now you're ready to start working on the project!

## How to Complete the Class

To complete the `MultiNodeCategoricalDecisionTree` class, you need to implement several key methods. Here's a guide on how to approach each method:

1. `_build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict[str, Any]`:
   - This is the core method that recursively builds the decision tree.
   - Implement the following logic:
     a. Check if the maximum depth has been reached or if the number of samples is less than `min_samples_split`.
     b. If either condition is true, create a leaf node with the majority class.
     c. If not, find the best split using the `_best_split` method.
     d. Create a decision node with the best feature and split point.
     e. Split the data and recursively build subtrees for each split.
   - Return a dictionary representing the node structure.

2. `_best_split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]`:
   - Implement the logic to find the best feature and split point for a given node.
   - For each feature:
     a. Find unique values in the feature.
     b. For each unique value, calculate the information gain or Gini impurity.
     c. Keep track of the split that results in the highest information gain.
   - Return a dictionary containing the best feature, split point, and related information.

3. `_calculate_feature_importances(self) -> np.ndarray`:
   - Traverse the tree and calculate feature importances based on the reduction in impurity at each split.
   - Normalize the importances so they sum to 1.
   - Return an array of feature importances.

4. `_predict_single(self, x: np.ndarray) -> Any`:
   - Implement the logic to traverse the tree for a single sample and return the predicted class.
   - Start at the root node and follow the appropriate branch based on the feature values until reaching a leaf node.
   - Return the majority class of the leaf node.

5. `_predict_proba_single(self, x: np.ndarray) -> np.ndarray`:
   - Similar to `_predict_single`, but instead of returning the majority class, return the class probabilities.
   - The probabilities should be based on the distribution of classes in the leaf node.
   - Return an array of probabilities for each class.

## Project Structure

The project includes two main Jupyter notebooks for practical implementation:

1. **Titanic Challenge Notebook** (`titanic/main-titanic.ipynb`):
   - This notebook provides a basic framework for the Titanic dataset challenge, including data loading, preprocessing, exploratory data analysis (EDA), model training, and evaluation.

2. **COVID-19 Challenge Notebook** (`corona/main-corona.ipynb`):
   - This notebook focuses on analyzing COVID-19 testing data for trend analysis and policy-making, with similar sections for data loading, preprocessing, and model evaluation.

## Additional Tips
- Use numpy operations for efficiency whenever possible.
- Make sure to handle edge cases, such as empty nodes or features with only one unique value.
- Consider adding helper methods for calculating impurity (e.g., Gini impurity or entropy) and for splitting the data.
- Test your implementation thoroughly with different datasets and compare results with scikit-learn's DecisionTreeClassifier.

By completing these methods, you will have a fully functional multi-node categorical decision tree classifier that can be used with scikit-learn's cross-validation and evaluation tools.