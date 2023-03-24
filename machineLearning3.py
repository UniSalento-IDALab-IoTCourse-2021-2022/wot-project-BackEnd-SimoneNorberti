import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump, load

# 1. Allenare e salvare il modello Isolation Forest utilizzando scikit-learn:
print("Start")
file = 'data.csv'
df = pd.read_csv(file, delimiter=',')
df = df.dropna()  # Dropping missing records
attributes = ['Gait_Speed', 'Grip_Strength', 'Muscle mass']
X_train = df[attributes]

clf = IsolationForest(random_state=0).fit(X_train)
dump(clf, 'isolation_forest.joblib')

# 2. Utilizzare la libreria joblib per caricare il modello salvato e convertirlo in un modello TensorFlow:

clf = load('isolation_forest.joblib')

# Creare un grafo TensorFlow
graph_def = tf.compat.v1.GraphDef()
with tf.Graph().as_default() as graph:
    # Creare un placeholder per i dati di input
    input_data = tf.compat.v1.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='input_data')
    # Creare un nodo TensorFlow per il modello Isolation Forest
    score_op = tf.compat.v1.py_func(clf.decision_function, [input_data], tf.float64, name='score_op')
    # Aggiungere il nodo al grafo
    graph_def.node.extend([score_op.op.node_def])

# Inizializzare le variabili del grafo
with tf.compat.v1.Session(graph=graph) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    # Salvare il grafo in formato protobuf
    tf.io.write_graph(graph_or_graph_def=sess.graph_def, logdir='.', name='isolation_forest.pb', as_text=False)

# 3. Convertire il modello TensorFlow in un modello TFLite:

# Caricare il grafo salvato
with tf.compat.v1.gfile.GFile('isolation_forest.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())

# Convertire il grafo in un modello TFLite
converter = tf.lite.TFLiteConverter.from_session(sess, [input_data], [score_op])
tflite_model = converter.convert()

# Salvare il modello TFLite su un file
with open('isolation_forest.tflite', 'wb') as f:
    f.write(tflite_model)
