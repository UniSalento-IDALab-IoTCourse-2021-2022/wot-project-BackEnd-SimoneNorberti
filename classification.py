# TODO
#           isolation forest
#           or Supervised Binary Classification:
#                                       Logistic Regression
#                                       k-Nearest Neighbors
#                                       Decision Trees
#                                       Support Vector Machine
#                                       Naive Bayes
#
# How often retrain the iForest?

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# IMPORT DATA FROM CSV FILE
file = 'data.csv'
df = pd.read_csv(file, delimiter=',')
print("len() before dropping missing records: {}".format(len(df)))
df = df.dropna()  # Dropping missing records
print("len() after dropping missing records: {}".format(len(df)))

print(df.sample(5))
print(df.describe())
attributes_extended = ['Sex', 'Age', 'BMI', 'Risk & Malnutrition', 'Gait_Speed', 'Grip_Strength', 'Muscle mass']
attributes_essential = ['Age', 'BMI', 'Gait_Speed', 'Grip_Strength', 'Muscle mass']

# PLOT PAIRWISE ATTRIBUTES
palette = ['#ff7f0e', '#1f77b4']
sns.pairplot(df, vars=attributes_essential, hue='Sarcopenia', palette=palette)
plt.show()

TRAIN_SIZE = 0.8
# X = df.drop(columns=['Sarcopenia']).copy()
X = df[attributes_essential]
y = df['Sarcopenia']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE)

print('X_train.shape: {}    X_test.shape: {}'.format(X_train.shape, X_test.shape))
print('y_train.shape: {}    y_test.shape: {}'.format(y_train.shape, y_test.shape))

SVM_flag = True  # Support Vector Machine
LR_flag = True  # Logistic Regression
test_flag = False  # True if u want evaluation stuff


# ------------------------------------------------------------------------------------------------

def k_fold_accuracy(n_splits=10, Xset=None, yset=None, model_to_evaluate=None):
    cv = KFold(n_splits=n_splits, random_state=1, shuffle=True)  # prepare the cross-validation procedure
    scores = cross_val_score(model_to_evaluate, Xset, yset, scoring='accuracy', cv=cv, n_jobs=-1)  # evaluate model
    print('K-Fold Accuracy: %.3f (%.3f)   (k=10)' % (np.mean(scores), np.std(scores)))  # report performance


def training_prediction(model=None):
    model.fit(X_train, y_train)  # Training
    y_predicted = model.predict(X_test)  # Testing
    print("Accuracy:", accuracy_score(y_test, y_predicted))
    print(classification_report(y_test, y_predicted, target_names=['No', 'Yes']))
    mat = confusion_matrix(y_test, y_predicted)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues",
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yes'])
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()


""""""""""""""""""""""""
"SUPPORT VECTOR MACHINE"
""""""""""""""""""""""""
if SVM_flag:
    print("\nSVM INITIALIZATION...")

    if test_flag:
        for c in range(1, 801, 5):
            for gamma in [0.0001, 0.0005, 0.001, 0.002,
                          0.005, 0.007, 0.01, 0.05, 0.1, 0.5]:
                model_SVC = SVC(kernel='rbf', C=c, gamma=gamma)
                cv = KFold(n_splits=10, random_state=1, shuffle=True)  # prepare the cross-validation procedure
                scores = cross_val_score(model_SVC, X, y, scoring='accuracy', cv=cv, n_jobs=-1)  # evaluate model
                if np.mean(scores) > 0.854 and np.std(scores) < 0.07:
                    print('C={} gamma={}'.format(c, gamma))
                    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))  # report performance

    # ------------------------------------------------------------------------------

    model_SVC = SVC(kernel='rbf', C=626, gamma=0.002)  # C=626 gamma=0.002
    # K-Fold cross-validation
    k_fold_accuracy(n_splits=10, Xset=X, yset=y, model_to_evaluate=model_SVC)
    # Training + Prediction
    training_prediction(model_SVC)

"""""""""""""""""""""
"LOGISTIC REGRESSION"
"""""""""""""""""""""
if LR_flag:
    print("\nLR INITIALIZATION...")
    model_LR = LogisticRegression()
    # K-Fold cross-validation
    k_fold_accuracy(n_splits=10, Xset=X, yset=y, model_to_evaluate=model_LR)
    # Training + Prediction
    training_prediction(model_LR)







"""
# Find best parameters---------------------------------------------------------
param_grid = {'C': [1, 2, 5, 10, 20, 30, 40, 50, 80, 100, 200, 500],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
              }
grid = GridSearchCV(model, param_grid)
grid.fit(X_train, y_train)
print("Best parameters: {}".format(grid.best_params_))
model = grid.best_estimator_
print(model)
"""