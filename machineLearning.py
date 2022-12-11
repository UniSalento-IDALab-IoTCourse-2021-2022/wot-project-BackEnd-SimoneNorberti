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

from sklearn2pmml import PMMLPipeline, sklearn2pmml


def train_send_model():
    # IMPORT DATA FROM CSV FILE
    file = 'data.csv'
    df = pd.read_csv(file, delimiter=',')
    print("len() before dropping missing records: {}".format(len(df)))
    df = df.dropna()  # Dropping missing records
    print("len() after dropping missing records: {}".format(len(df)))

    print(df.sample(5))
    print(df.describe())
    attributes_extended = ['Sex', 'Age', 'BMI', 'Risk & Malnutrition', 'Gait_Speed', 'Grip_Strength', 'Muscle mass']
    # attributes_essential = ['Age', 'BMI', 'Gait_Speed', 'Grip_Strength', 'Muscle mass']
    attributes_essential = ['Gait_Speed', 'Grip_Strength', 'Muscle mass']

    # PLOT PAIRWISE ATTRIBUTES
    palette = ['#ff7f0e', '#1f77b4']
    sns.pairplot(df, vars=attributes_essential, hue='Sarcopenia', palette=palette)
    plt.show()

    TRAIN_SIZE = 0.8
    X = df[attributes_essential]
    y = df['Sarcopenia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE)

    print('X_train.shape: {}    X_test.shape: {}'.format(X_train.shape, X_test.shape))
    print('y_train.shape: {}    y_test.shape: {}'.format(y_train.shape, y_test.shape))

    SVM_flag = True  # Support Vector Machine

    if SVM_flag:
        print("\nSVM INITIALIZATION...")

        model_SVC = SVC(kernel='rbf', C=626, gamma=0.002)
        pipeline = PMMLPipeline([("SVM Classifier", SVC(kernel='rbf', C=626, gamma=0.002))])
        pipeline.fit(X_train, y_train)
        sklearn2pmml(pipeline, "model_SVC2.pmml", with_repr=True)
        f = open("model_SVC2.pmml", "r")
        bigstring = f.read()
        # print(bigstring)
        return bigstring
        # TODO model for Andorid App Java


train_send_model()


'''

LR_flag = True  # Logistic Regression
KNN_flag = True  # K-Nearest Neighbor
GaussinNB_flag = True  # Gaussian Naive Bayes
DecisionTree_flag = True
ensemble_try = True
test_flag = False  # True if u want evaluation stuff


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
