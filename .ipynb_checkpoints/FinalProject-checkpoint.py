# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Loading & Partitioning Datasets (Iris, Breast Cancer, Wine)

# %%
import pandas as pd
import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()

Xi = iris.data

Yi = (iris.target > 1.5).reshape(-1,1).astype(float)
Yi[Yi==0] = -1

Xi_80train, Xi_20test, Yi_80train, Yi_20test = train_test_split(Xi, Yi, test_size=0.2, random_state=1)
Xi_50train, Xi_50test, Yi_50train, Yi_50test = train_test_split(Xi, Yi, test_size=0.5, random_state=1)
Xi_20train, Xi_80test, Yi_20train, Yi_80test = train_test_split(Xi, Yi, test_size=0.8, random_state=1)

breast_cancer = datasets.load_breast_cancer()

Xb = breast_cancer.data

Yb = (breast_cancer.target > 0.5).reshape(-1,1).astype(float)
Yb[Yb==0] = -1

Xb_80train, Xb_20test, Yb_80train, Yb_20test = train_test_split(Xb, Yb, test_size=0.2, random_state=1)
Xb_50train, Xb_50test, Yb_50train, Yb_50test = train_test_split(Xb, Yb, test_size=0.5, random_state=1)
Xb_20train, Xb_80test, Yb_20train, Yb_80test = train_test_split(Xb, Yb, test_size=0.8, random_state=1)

wine = datasets.load_wine()

Xw = wine.data

Yw = (wine.target > 1.5).reshape(-1,1).astype(float)
Yw[Yw==0] = -1

Xw_80train, Xw_20test, Yw_80train, Yw_20test = train_test_split(Xw, Yw, test_size=0.2, random_state=1)
Xw_50train, Xw_50test, Yw_50train, Yw_50test = train_test_split(Xw, Yw, test_size=0.5, random_state=1)
Xw_20train, Xw_80test, Yw_20train, Yw_80test = train_test_split(Xw, Yw, test_size=0.8, random_state=1)
print(Xi_50train.shape)
print(Xi_50test.shape)
print(Xw.shape)


# %%
def vis(X, Y, W=None, b=None):
    indices_neg1 = (Y == -1).nonzero()[0]
    indices_pos1 = (Y == 1).nonzero()[0]
    plt.scatter(X[:,0][indices_neg1], X[:,1][indices_neg1],
    c='blue', label='class -1')
    plt.scatter(X[:,0][indices_pos1], X[:,1][indices_pos1],
    c='red', label='class 1')
    plt.legend()
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    if W is not None:
        w0 = W[0]
        w1 = W[1]
        temp = -w1*np.array([X[:,1].min(), X[:,1].max()])/w0-b/w0
        x0_min = max(temp.min(), X[:,0].min())
        x0_max = min(temp.max(), X[:,1].max())
        x0 = np.linspace(x0_min,x0_max,100)
        x1 = -w0*x0/w1-b/w1
        plt.plot(x0,x1,color='black')
        
    plt.show()


vis(Xi_80train, Yi_80train)


# %% [markdown]
# # Training Datasets - SVM with the RBF Kernel

# %%
def calc_error(X, Y, classifier):
    Y_pred = classifier.predict(X)
    e = 1 - accuracy_score(Y, Y_pred)
    return e

def draw_heatmap(training_errors, gamma_list, C_list):
    plt.figure(figsize = (5,4))
    ax = sns.heatmap(training_errors, annot=True, fmt='.3f',
        xticklabels=gamma_list, yticklabels=C_list)
    ax.collections[0].colorbar.set_label("error")
    ax.set(xlabel = r'$\gamma$', ylabel=r'$C$')
    plt.title(r'Training error w.r.t $C$ and $\gamma$')
    plt.show()

def svm_rbf_training(X_train,Y_train):
    C_list = [1, 10, 100, 1000, 10000]
    gamma_list = [1e-6, 1e-5, 1e-4, 1e-3,1e-2]
    
    opt_e_training = 1.0 # Optimal training error.
    opt_accuracy = 0.0 # Optimal validation accuracy
    opt_classifier = None # Optimal classifier.
    opt_C = None # Optimal C.
    opt_gamma = None # Optimal C.
    # Training errors
    training_errors = np.zeros((len(C_list), len(gamma_list)))
    for i, C in enumerate(C_list):
        for j, gamma in enumerate(gamma_list):
            # Create a SVM classifier with RBF kernel.
            classifier = svm.SVC(kernel='rbf', C=C, gamma=gamma)
            # Use the classifier to fit the training set (use X_train, Y_train).
            avg_accuracy, avg_training_error = k_fold(X_train, Y_train, classifier)
            training_errors[i, j] = avg_training_error

            if avg_accuracy > opt_accuracy:
                opt_accuracy = avg_accuracy 
                opt_e_training = avg_training_error
                opt_classifier = classifier
                opt_C = C
                opt_gamma = gamma

    print("Optimal parameters: " + str(opt_C) + str(opt_gamma))
    draw_heatmap(training_errors, gamma_list, C_list)
    
    final_model = svm.SVC(kernel='rbf', C=opt_C, gamma=opt_gamma)
    final_model.fit(X_train, Y_train)

    return [final_model, opt_C, opt_gamma]

# Shuffle & create 3 folds

def k_fold(X, Y, model, n_splits=3):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    fold_accuracies = []
    fold_training_errors = []
    all_fold_accuracies = []
    all_fold_training_errors = []

    for train_index, val_index in kfold.split(X):
        # Split the data into training and validation sets for each fold
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        
        model.fit(X_train, Y_train)  # Train the model on 2/3 of the data
        Y_pred_val = model.predict(X_val)  # Validate on 1/3 of the data
        
        # Calculate validation accuracy and training error
        accuracy = accuracy_score(Y_val, Y_pred_val)
        fold_accuracies.append(accuracy)
        training_error = calc_error(X_train, Y_train, model)
        fold_training_errors.append(training_error)
        all_fold_accuracies.append(accuracy)
        all_fold_training_errors.append(training_error)
    
    # Return average accuracy and training error
    avg_accuracy = np.mean(fold_accuracies)
    avg_training_error = np.mean(fold_training_errors)

    # Plot the curves (accuracy and error) during cross-validation
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(all_fold_accuracies, label='Validation Accuracy')
    # plt.xlabel('Fold')
    # plt.ylabel('Accuracy')
    # plt.title('Validation Accuracy per Fold')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(all_fold_training_errors, label='Training Error')
    # plt.xlabel('Fold')
    # plt.ylabel('Training Error')
    # plt.title('Training Error per Fold')
    # plt.tight_layout()
    # plt.show()
    
    return avg_accuracy, avg_training_error

opt_i80_results = svm_rbf_training(Xi_80train, Yi_80train.ravel())
test_accuracy = opt_i80_results[0].score(Xi_20test, Yi_20test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_i50_results = svm_rbf_training(Xi_50train, Yi_50train.ravel())
test_accuracy = opt_i50_results[0].score(Xi_50test, Yi_50test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_i20_results = svm_rbf_training(Xi_20train, Yi_20train.ravel())
test_accuracy = opt_i20_results[0].score(Xi_80test, Yi_80test)
print(f"Test accuracy: {test_accuracy:.4f}")

opt_w80_results = svm_rbf_training(Xw_80train, Yw_80train.ravel())
test_accuracy = opt_w80_results[0].score(Xw_20test, Yw_20test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_w50_results = svm_rbf_training(Xw_50train, Yw_50train.ravel())
test_accuracy = opt_w50_results[0].score(Xw_50test, Yw_50test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_w20_results = svm_rbf_training(Xw_20train, Yw_20train.ravel())
test_accuracy = opt_w20_results[0].score(Xw_80test, Yw_80test)
print(f"Test accuracy: {test_accuracy:.4f}")

opt_b80_results = svm_rbf_training(Xb_80train, Yb_80train.ravel())
test_accuracy = opt_b80_results[0].score(Xb_20test, Yb_20test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_b50_results = svm_rbf_training(Xb_50train, Yb_50train.ravel())
test_accuracy = opt_b50_results[0].score(Xb_50test, Yb_50test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_b20_results = svm_rbf_training(Xb_20train, Yb_20train.ravel())
test_accuracy = opt_b20_results[0].score(Xb_80test, Yb_80test)
print(f"Test accuracy: {test_accuracy:.4f}")


# %%
from functools import partial
import scipy
from matplotlib.colors import ListedColormap

# Visualization function
def vis(X, Y, model):
    if model is not None:
        h = .02  # Step size in meshgrid
        x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        x0s, x1s = np.meshgrid(np.arange(x0_min, x0_max, h),
                               np.arange(x1_min, x1_max, h))
        xs = np.stack([x0s, x1s], axis=-1).reshape(-1, 2)

        # Predict class using the model and data
        ys_pred = model.predict(xs)  # Use model.predict instead of calling the classifier
        ys_pred = ys_pred.reshape(x0s.shape)

        # Put the result into a color plot.
        cmap_light = ListedColormap(['#00AAFF', '#FFAAAA'])
        plt.pcolormesh(x0s, x1s, ys_pred, cmap=cmap_light, alpha=0.3)

    indices_neg1 = (Y == -1).nonzero()[0]
    indices_pos1 = (Y == 1).nonzero()[0]
    plt.scatter(X[indices_neg1, 0], X[indices_neg1, 1],
                c='blue', label='class -1', alpha=0.3)
    plt.scatter(X[indices_pos1, 0], X[indices_pos1, 1],
                c='red', label='class +1', alpha=0.3)
    plt.legend()
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.show()

def knn_training(X_train, Y_train):
    k_list = [1, 3, 5, 7, 9]
    opt_val_error = 1.0
    opt_k = None
    opt_knn_classifier = None

    for k in k_list:
        print(f"k = {k}")
        
        # Use scikit-learn's KNeighborsClassifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k)

        knn_classifier.fit(X_train, Y_train)
        
        # Visualize the decision boundary
        vis(X_train, Y_train, knn_classifier)

        # Perform cross-validation
        avg_accuracy, avg_training_error = k_fold(X_train, Y_train, knn_classifier)
        
        print(f"Validation Accuracy: {avg_accuracy:.4f}")
        print(f"Validation Error: {avg_training_error:.4f}\n")
        
        # Select optimal k based on validation error
        if avg_training_error < opt_val_error:
            opt_val_error = avg_training_error
            opt_k = k
            opt_knn_classifier = knn_classifier

    print(f"Optimal k: {opt_k}")
    return opt_knn_classifier


opt_i80_results = knn_training(Xi_80train[:,[3,1]], Yi_80train.ravel())
test_accuracy = opt_i80_results.score(Xi_20test[:,[3,1]], Yi_20test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_i50_results = knn_training(Xi_50train[:,[3,1]], Yi_50train.ravel())
test_accuracy = opt_i50_results.score(Xi_50test[:,[3,1]], Yi_50test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_i20_results = knn_training(Xi_20train[:,[3,1]], Yi_20train.ravel())
test_accuracy = opt_i20_results.score(Xi_80test[:,[3,1]], Yi_80test)
print(f"Test accuracy: {test_accuracy:.4f}")

opt_w80_results = knn_training(Xw_80train[:,[3,1]], Yw_80train.ravel())
test_accuracy = opt_w80_results.score(Xw_20test[:,[3,1]], Yw_20test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_w50_results = knn_training(Xw_50train[:,[3,1]], Yw_50train.ravel())
test_accuracy = opt_w50_results.score(Xw_50test[:,[3,1]], Yw_50test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_w20_results = knn_training(Xw_20train[:,[3,1]], Yw_20train.ravel())
test_accuracy = opt_w20_results.score(Xw_80test[:,[3,1]], Yw_80test)
print(f"Test accuracy: {test_accuracy:.4f}")

opt_b80_results = knn_training(Xb_80train[:,[3,1]], Yb_80train.ravel())
test_accuracy = opt_b80_results.score(Xb_20test[:,[3,1]], Yb_20test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_b50_results = knn_training(Xb_50train[:,[3,1]], Yb_50train.ravel())
test_accuracy = opt_b50_results.score(Xb_50test[:,[3,1]], Yb_50test)
print(f"Test accuracy: {test_accuracy:.4f}")
opt_b20_results = knn_training(Xb_20train[:,[3,1]], Yb_20train.ravel())
test_accuracy = opt_b20_results.score(Xb_80test[:,[3,1]], Yb_80test)
print(f"Test accuracy: {test_accuracy:.4f}")

# %%

# %%

# %%
