#!/usr/bin/python3.7
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import database
import json
import time
import smtpfile
import pymongo
from sklearn.ensemble import IsolationForest
import sys
print("Python version", sys.version)

app = Flask(__name__)
CORS(app)


def majority_vote(arr):
    """
    Restituisce l'elemento che compare più spesso nell'array
    """
    # Inizializzazione del dizionario vuoto per contare le occorrenze degli elementi
    count = {}

    # Conteggio delle occorrenze di ogni elemento nell'array
    for el in arr:
        if el in count:
            count[el] += 1
        else:
            count[el] = 1

    # Trovare l'elemento con il conteggio più alto
    max_count = 0
    max_el = None
    for el, cnt in count.items():
        if cnt > max_count:
            max_count = cnt
            max_el = el

    # Restituisce l'elemento più comune
    return max_el


@app.route('/api/listapazienti')
def get_pazienti():
    # TODO ricevere lista pazienti con query MongoDB
    objects = [
        {'ID': 'ID001', 'nome': 'Mario', 'cognome': 'Rossi'},
        {'ID': 'ID002', 'nome': 'Giuseppe', 'cognome': 'Verdi'},
        {'ID': 'ID003', 'nome': 'Francesco', 'cognome': 'Bianchi'}
    ]
    return jsonify(objects)


@app.route('/api/pazienti/<string:object_id>')
def get_data(object_id):
    print("ObjectID: " + object_id)
    # Configura la connessione al database MongoDB
    client = pymongo.MongoClient("mongodb+srv://snorb:7OGFhqrLw8rTfCaL@clustersarcopenia0.gvzw6w6.mongodb.net/"
                                 "?retryWrites=true&w=majority")
    db = client["SarcopeniaDB"]
    collection = db["measurement"]

    # Crea la query
    query = {"ID": object_id}

    # Esegue la query e ottiene i risultati
    results = collection.find(query)

    data = []
    for result in results:
        result['_id'] = str(result['_id'])
        data.append(result)
    print(data)

    return jsonify(data)


@app.route('/')
def hello_world():
    return 'Sarcopenia Project - Norberti Simone [IoT]'


@app.route('/api/endsession', methods=['POST'])
def end_session():
    """
        INPUT: ID
    """

    end_data = request.get_json()
    print('/api/endsession  ID:', end_data)

    DB = 'SarcopeniaDB'
    MEASUREMENT = 'measurement'
    ID = end_data["ID"]

    # -------------    ANTI-DOUBLE REQUEST     -------------
    try:
        database.db_get_all_to_predict(DB, MEASUREMENT, ID)
    except:
        return '{ "ML success": -1 }'


    # 0: controllo se esiste il modello
    model = database.db_get_model(DB, ID)
    if model is None:
        print("Model not found: Check the number of measurements in db...")

        # 1: se il modello esiste e vi sono abbastanza misure, fare ANOMALY DETECTION
        if database.db_count(DB, MEASUREMENT, ID) >= 30:  # se misurazioni <30 il modello non viene creato
            print("Model not found: Training new model...")

            dataset = database.db_get_all(DB, MEASUREMENT, ID)
            new_model = IsolationForest(contamination=0.1)  # random_state=0, contamination=0.01
            new_model.fit(dataset)  # Train the model
            database.db_insert_model(DB, new_model, ID)
            database.db_reset_anomaly(DB, MEASUREMENT, ID)
            print("Model not found: Model trained!")

            return '{ "ML success": 2 }'  # model trained, no anomaly detection
        print("Not enaught measurements to fit the model")
        return '{ "ML success": 1 }'  # model not trained, no anomaly detection

    # 2: il modello esiste, facciamo ANOMALY DETECTION
    data_to_predict = database.db_get_all_to_predict(DB, MEASUREMENT, ID)  # anomaly = -1 (to check)
    database.db_reset_anomaly(DB, MEASUREMENT, ID)
    print("\nSession's measurements:\n", data_to_predict)
    pred = model.predict(data_to_predict)  # Prediction
    print("\nPredictions:", pred)

    # 3: Majority voting per rilevare anomalia
    print("\nMajority voting...")
    majority_result = majority_vote(pred)
    if majority_result == -1:
        print("Anomaly detected!!!")
        # TODO prendere email medico + altre info paziente dal DB
        smtpfile.send_email(ID)
    elif majority_result == 1:
        print("No anomaly")
    else:
        print("[Error] Majority voting error")

    # 3: ELIMINARE vecchio modello, ANOMALY = 0, ALLENARE nuovo modello
    print("\nRe-training the model with the new measurements...")
    database.db_delete_model(DB, ID)
    dataset = database.db_get_all(DB, MEASUREMENT, ID)
    print("Total measurement:", dataset.shape, "(#meas, #attr)")

    new_model = IsolationForest(contamination=0.1)  # random_state=0, contamination=0.01
    new_model.fit(dataset)
    database.db_insert_model(DB, new_model, ID)
    print("Re-train end.")

    return '{ "ML success": 0 }'


@app.route('/api/senddata', methods=['GET', 'POST'])
def data_receiver():
    """
    paziente --> codice_fiscale (o ID)
    misurazioni --> (no) FM, FFM, acc_x, acc_y, acc_z, muscle_strenght (mai, al massimo sarà lo smartphone a combinarli)
                --> (si) GAIT_SPEED, GRIP_STRENGHT, MUSCLE_MASS
    """

    ''' DATA RECEIVED BY REST POST '''
    data = request.get_json()
    data['ANOMALY'] = -1  # Not checked
    print('Data received: {}'.format(data))

    DB = 'SarcopeniaDB'
    MEASUREMENT = 'measurement'
    MEASUREMENT_TEMP = 'measurement_temp'

    ''' inserire nuovo dato nel DB '''
    database.db_insert_one(DB, MEASUREMENT, data)

    return '{ "data success": 0 }'


if __name__ == '__main__':
    app.run(host='0.0.0.0')
