
# How often retrain the classificators?

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

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
KNN_flag = True  # K-Nearest Neighbor
GaussinNB_flag = True  # Gaussian Naive Bayes
DecisionTree_flag = True
test_flag = False  # True if u want evaluation stuff


# ------------------------------------------------------------------------------------------------

def k_fold_accuracy(n_splits=10, Xset=None, yset=None, model_to_evaluate=None):
    cv = KFold(n_splits=n_splits, random_state=1, shuffle=True)  # prepare the cross-validation procedure
    scores = cross_val_score(model_to_evaluate, Xset, yset, scoring='accuracy', cv=cv, n_jobs=-1)  # evaluate model
    print('K-Fold Accuracy: %.3f (%.3f)   (k=10)' % (np.mean(scores).__float__(), np.std(scores).__float__()))  #
    # report performance


def training_prediction(model=None, X_train=None, y_train=None):
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


if SVM_flag:
    print("\nSVM INITIALIZATION...")

    model_SVC = SVC(kernel='rbf', C=626, gamma=0.002)  # C=626 gamma=0.002 --> best parameters found
    k_fold_accuracy(n_splits=10, Xset=X, yset=y, model_to_evaluate=model_SVC)  # K-Fold cross-validation
    training_prediction(model_SVC, X_train, y_train)  # Training + Prediction

if LR_flag:
    print("\nLR INITIALIZATION...")

    model_LR = LogisticRegression()
    k_fold_accuracy(n_splits=10, Xset=X, yset=y, model_to_evaluate=model_LR)  # K-Fold cross-validation
    training_prediction(model_LR, X_train, y_train)  # Training + Prediction

if KNN_flag:
    print("\nKNN INITIALIZATION...")

    model_KNN = KNeighborsClassifier(n_neighbors=13)
    k_fold_accuracy(n_splits=10, Xset=X, yset=y, model_to_evaluate=model_KNN)  # K-Fold cross-validation
    training_prediction(model_KNN, X_train, y_train)  # Training + Prediction

if GaussinNB_flag:
    print("\nGaussianNB INITIALIZATION...")

    model_GNB = GaussianNB()
    k_fold_accuracy(n_splits=10, Xset=X, yset=y, model_to_evaluate=model_GNB)  # K-Fold cross-validation
    training_prediction(model_GNB, X_train, y_train)  # Training + Prediction

if DecisionTree_flag:
    print("\nDecisionTree INITIALIZATION...")

    model_DTC = DecisionTreeClassifier()
    k_fold_accuracy(n_splits=10, Xset=X, yset=y, model_to_evaluate=model_DTC)  # K-Fold cross-validation
    training_prediction(model_DTC, X_train, y_train)  # Training + Prediction

'''
ENSEMBLE TRY
'''

pred = []

pred.append(model_SVC.predict(X_test))
pred.append(model_LR.predict(X_test))
pred.append(model_KNN.predict(X_test))
pred.append(model_GNB.predict(X_test))
pred.append(model_DTC.predict(X_test))
print("SVC={} \nLR={} \nKNN={} \nGNB={} \nDTC={}".format(pred[0], pred[1], pred[2], pred[3], pred[4]))

'''
MAJORITY VOTING
'''
m = None  # winner
len_pred = len(pred[0])
n_classifier = 5
print(len_pred)
final = []

for j in range(len_pred):
    score = 0
    for i in [0, 1, 2, 3, 4]:  # range(n_classifier)
        print("i,j = {},{}".format(i, j))
        if i == 0:
            m = pred[i][j]
            print('m = ', m)
            score = 1
        elif pred[i][j] == m:
            score += 1
        else:
            if score == 0:
                m = pred[i][j]
                continue
            score -= 1
    final.append(m)
    print(final[j])

print("Ensamble Accuracy:", accuracy_score(y_test, final))
# TODO complete the esamble learning
"""
Initialize an element m and a counter i with i = 0
For each element x of the input sequence:
    If i = 0, then assign m = x and i = 1
    else if m = x, then assign i = i + 1
    else assign i = i − 1
Return m
"""

"""
# after first print... in SVM
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
"""

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
