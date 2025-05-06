import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
import time

# Load the dataset from a file
def load_dataset(filename):
    try:
        data = np.loadtxt(filename)        # load the data
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

    X = data[:, 1:]             # Features (all columns except the first)
    y = data[:, 0]              # Class labels (first column)
    return X, y

# Evaluation function using k-fold cross-validation
def evaluate_subset(X_subset, y, cv_splits=5):
    clf = KNeighborsClassifier(n_neighbors=1)           # 1 nearest neighbor
    kf = KFold(n_splits=cv_splits)                      # 5-fold
    scores = cross_val_score(clf, X_subset, y, cv=kf)       # 5-fold cross-validation
    return scores.mean()            # return the mean accuracy

# Forward selection method
def forward_selection(X, y, cv_splits=5):
    start_time = time.time()                   # start time
    num_features = X.shape[1]                  # number of features
    best_subset = []                           # best feature subset
    best_accuracy = 0.0                        # best accuracy

    # Loop through all features
    while len(best_subset) < num_features:
        candidate_accuracy = []                 # list of candidate accuracies
        candidate_features = []                 # list of candidate feature sets

        # Generate the list of indices not in best_subset
        remaining_features = []
        for i in range(num_features):
            if i not in best_subset:
                remaining_features.append(i)

        # Evaluate each subset directly in the loop
        for i in remaining_features:
            subset = best_subset + [i]             # add a new feature to the current best subset
            accuracy = evaluate_subset(X[:, subset], y, cv_splits)
            candidate_accuracy.append(accuracy)
            candidate_features.append(subset)
            print(f"Using feature(s) {subset} accuracy is {accuracy:.1%}")

        # Find the best candidate
        max_accuracy = max(candidate_accuracy)
        max_index = candidate_accuracy.index(max_accuracy)
        best_candidate = candidate_features[max_index]

        # Update the best subset and accuracy
        if max_accuracy > best_accuracy:
            best_accuracy = max_accuracy
            best_subset = best_candidate
            print(f"Feature set {best_subset} was best, accuracy is {best_accuracy:.1%}\n")
        else:
            break

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds\n")
        
    return best_subset, best_accuracy

# Backward elimination method
def backward_elimination(X, y, cv_splits=5):
    start_time = time.time()                                # start time
    num_features = X.shape[1]                               # number of features
    best_subset = list(range(num_features))                 # best feature subset
    best_accuracy = evaluate_subset(X, y, cv_splits)        # best accuracy

    # Loop through all features
    while len(best_subset) > 0:
        candidate_accuracy = []                             # list of candidate accuracies
        candidate_features = []                             # list of candidate feature sets

        # Generate the list of remaining features
        remaining_features = best_subset.copy()

        # Evaluate each subset directly in the loop
        for i in remaining_features:
            subset = remaining_features.copy()              # copy the current best subset
            subset.remove(i)                                # remove a feature from the current best subset
            accuracy = evaluate_subset(X[:, subset], y, cv_splits)
            candidate_accuracy.append(accuracy)             
            candidate_features.append(subset)
            print(f"Using feature(s) {subset} accuracy is {accuracy:.1%}")

        # Find the best candidate
        max_accuracy = max(candidate_accuracy)
        max_index = candidate_accuracy.index(max_accuracy)
        best_candidate = candidate_features[max_index]

        # Update the best subset and accuracy
        if max_accuracy > best_accuracy:
            print(f"Feature set {best_candidate} was best, accuracy is {max_accuracy:.1%}\n")
            best_accuracy = max_accuracy
            best_subset = best_candidate
        else:
            break

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds\n")
        
    return best_subset, best_accuracy

# Main function
if __name__ == "__main__":
    print("Welcome to Justin Chiu's Feature Selection Algorithm\n")
    filename = input("Type in the name of the file to test: ")
    X, y = load_dataset(filename)
    
    algorithms = {
        "1": "Forward Selection",
        "2": "Backward Elimination"
    }
    
    selection = input("Type the number of the algorithm you want to run. \n\n" +
                     "1) Forward Selection\n" +
                     "2) Backward Elimination\n\n")
    
    if selection not in algorithms:
        raise ValueError("Invalid algorithm selection.")
    
    print(f"\nThis dataset has {X.shape[1]} features (not including the class attribute), with {X.shape[0]} instances.")
    
    accuracy_all_features = evaluate_subset(X, y)
    print(f"Running nearest neighbor with all {X.shape[1]} features, using {algorithms[selection]} evaluation, I get an accuracy of {accuracy_all_features:.1%}.\n")

    if selection == "1":
        best_subset, best_accuracy = forward_selection(X, y)
    elif selection == "2":
        best_subset, best_accuracy = backward_elimination(X, y)

    print(f"\nFinished search!! The best feature subset is {best_subset}, which has an accuracy of {best_accuracy:.1%}.\n")
