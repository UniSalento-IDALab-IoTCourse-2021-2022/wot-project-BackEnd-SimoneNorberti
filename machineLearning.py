import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def train_model():
    # IMPORT DATA FROM CSV FILE
    file = 'data.csv'
    df = pd.read_csv(file, delimiter=',')
    df = df.dropna()  # Dropping missing records

    # print(df.sample(5))
    # print(df.describe())

    attributes = ['Gait_Speed', 'Grip_Strength', 'Muscle mass']

    # PLOT PAIRWISE ATTRIBUTES
    palette = ['#ff7f0e', '#1f77b4']
    sns.pairplot(df, vars=attributes, hue='Sarcopenia', palette=palette)
    plt.show()

    X = df[attributes]

    model_IF = IsolationForest(random_state=0)
    model_IF.fit(X)
    X['anomaly'] = model_IF.predict(X)

    # Creazione di una figura 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Creazione del grafico a dispersione (scatter plot)
    ax.scatter(X['Gait_Speed'], X['Grip_Strength'], X['Muscle mass'], c=X['anomaly'], cmap='viridis')

    # Etichettatura degli assi
    ax.set_xlabel('Gait_Speed')
    ax.set_ylabel('Grip_Strength')
    ax.set_zlabel('Muscle mass')

    # Visualizzazione del grafico
    plt.show()


train_model()
