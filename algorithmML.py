# TODO
#           isolation forest
#           or Supervised Binary Classification:
#                                       Logistic Regression
#                                       k-Nearest Neighbors
#                                       Decision Trees
#                                       Support Vector Machine
#                                       Naive Bayes
#
# How often retain the iForest?

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# IMPORT DATA FROM CSV FILE
file = 'data.csv'
df = pd.read_csv(file, delimiter=',')
print("len before dropping missing records: {}".format(len(df)))
# Dropping missing records
df = df.dropna()
print("len after dropping missing records: {}".format(len(df)))

print(df.sample(5))
print(df.describe())
attributes_extended = ['Sex', 'Age', 'BMI', 'Risk & Malnutrition', 'Gait_Speed', 'Grip_Strength', 'Muscle mass']
attributes_essential = ['Age', 'BMI', 'Gait_Speed', 'Grip_Strength', 'Muscle mass']

# PLOT PAIRWISE ATTRIBUTES
palette = ['#ff7f0e', '#1f77b4']
sns.pairplot(df, vars=attributes_extended, hue='Sarcopenia', palette=palette)
# plt.title("Sarcopenia - using all attributes")
plt.show()
sns.pairplot(df, vars=attributes_essential, hue='Sarcopenia', palette=palette)
# plt.title("Sarcopenia - using selected attributes")
plt.show()

TRAIN_SIZE = 0.8
X = df.drop(columns=['Sarcopenia']).copy()
y = df['Sarcopenia']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE)

print('X_train.shape: {}'.format(X_train.shape)), print('y_train.shape: {}'.format(y_train.shape))
print('X_test.shape: {}'.format(X_test.shape)), print('y_test.shape: {}'.format(y_test.shape))

"""""""""""""""""""""
"LOGISTIC REGRESSION"
"""""""""""""""""""""


""""""""""""""""""
"ISOLATION FOREST"
""""""""""""""""""
# TODO anomaly detection must be performed for data of one single patient, not

# contamination : how much of the overall data we expect to be considered as an outlier.
#                 we can pass in a value between 0 and 0.5 or set it to auto.
# random_state: control the random selection process for splitting trees (seed).
CONTAMINATION = float(0.1)
RANDOM_STATE = 42

# DEFINE THE MODEL
model_IF = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)

# TRAINING
# model_IF.fit(df[attributes_essential])
model_IF.fit(X_train[attributes_essential])

# TESTING
print(X_test.sample(3))
X_test['anomaly_scores'] = model_IF.decision_function(X_test[attributes_essential])
X_test['anomaly'] = model_IF.predict(X_test[attributes_essential])

print(X_test.loc[:, ['ID', 'anomaly_scores', 'anomaly']])



# PLOTTING OUTLIERS
def outlier_plot(data, outlier_method_name, x_var, y_var,
                 xaxis_limits=[0, 1], yaxis_limits=[0, 1]):
    print(f'Outlier Method: {outlier_method_name}')

    # Create a dynamic title based on the method
    method = f'{outlier_method_name}_anomaly'

    # Print out key statistics
    print(f"Number of anomalous values {len(data[data['anomaly'] == -1])}")
    print(f"Number of non anomalous values  {len(data[data['anomaly'] == 1])}")
    print(f'Total Number of Values: {len(data)}')

    # Create the chart using seaborn
    g = sns.FacetGrid(data, col='anomaly', height=4, hue='anomaly', hue_order=[1, -1])
    g.map(sns.scatterplot, x_var, y_var)
    g.fig.suptitle(f'Outlier Method: {outlier_method_name}', y=1.10, fontweight='bold')
    g.set(xlim=xaxis_limits, ylim=yaxis_limits)
    axes = g.axes.flatten()
    axes[0].set_title(f"Outliers\n{len(data[data['anomaly'] == -1])} points")
    axes[1].set_title(f"Inliers\n {len(data[data['anomaly'] == 1])} points")
    return g


outlier_plot(X_test, 'Isolation Forest', 'Age', 'Muscle mass',
             [X_test['Age'].min()-1., X_test['Age'].max()+1],
             [X_test['Muscle mass'].min()-1., X_test['Muscle mass'].max()+1])
plt.show()