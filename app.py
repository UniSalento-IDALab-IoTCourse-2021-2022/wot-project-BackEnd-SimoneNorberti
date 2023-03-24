import pandas as pd
from flask import Flask, request, jsonify
import uuid
import database
import json
import time
from sklearn.ensemble import IsolationForest

app = Flask(__name__)


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


@app.route('/')
def hello_world():
    return 'Sarcopenia Project - Norberti Simone [IoT]'


@app.route('/api/endsession', methods=['POST'])
def end_session():
    """
    paziente --> codice fiscale (o ID)
    """

    end_data = request.get_json()
    print('Data received: {}'.format(end_data))

    DB = 'SarcopeniaDB'
    MEASUREMENT = 'measurement'
    MEASUREMENT_TEMP = 'measurement_temp'
    ID = end_data["ID"]

    # -------------    ANTI-DOUBLE REQUEST     -------------
    if database.db_count(DB, MEASUREMENT_TEMP, ID) == 0:
        print("---------------ANTI DOPPIA RICHIESTA---------------")
        time.sleep(5)
        return '{ "ML success": -1 }'
    # ------------------------------------------------------

    # 0: controllo se esiste il modello
    model = database.db_get_model(DB, ID)
    if model is None:
        #TODO sostituire MEASUREMENT_TEMP con anomaly -1,0,1
        database.db_delete_all(DB, MEASUREMENT_TEMP, ID)
        print("[MODEL NOT FOUND] - Controllo numero di misurazioni...")
        # 1: se il modello esiste e vi sono abbastanza misure, fare ANOMALY DETECTION
        if database.db_count(DB, MEASUREMENT, ID) >= 30:
            print("[MODEL NOT FOUND] -  Creazione modello in corso...")

            dataset = database.db_get_all(DB, MEASUREMENT, ID)

            new_model = IsolationForest(contamination=0.1)  # random_state=0, contamination=0.01
            new_model.fit(dataset)
            database.db_insert_model(DB, new_model, ID)
            print("[MODEL CREATED]")
            # database.db_delete_all(DB, MEASUREMENT_TEMP, ID)

        return '{ "ML success": 0 }'  # ritorna, no anomaly detection

    # 2: il modello esiste, facciamo ANOMALY DETECTION
    data_to_predict = database.db_get_all(DB, MEASUREMENT_TEMP, ID)

    print("\nMisurazioni della sessione: {}".format(data_to_predict))
    # print(data_to_predict)

    pred = model.predict(data_to_predict)
    #TODO Aggiungere attributo "ANOMALY"

    # print("\nLe predizioni sono: {}".format(pred))

    # 3: Majority voting per rilevare anomalia
    print("\nMajority voting...")
    majority_result = majority_vote(pred)
    if majority_result == -1:
        print("Anomalia Rilevata - Prediction per ogni misurazione: {}".format(pred))
        # TODO AVVISARE IL MEDICO
    elif majority_result == 1:
        print("No anomalia - Prediction per ogni misurazione: {}".format(pred))
    else:
        print("[ERROR] - Majority voting error!!!")

    # 3: ELIMINARE vecchio modello, ALLENARE nuovo modello
    print("\n[RE-TRAIN MODEL] - Start!")
    database.db_delete_all(DB, MEASUREMENT_TEMP, ID)
    database.db_delete_model(DB, ID)
    dataset = database.db_get_all(DB, MEASUREMENT, ID)
    '''
    dataset = pd.DataFrame.from_dict({
        'GAIT_SPEED': dataset["GAIT_SPEED"],
        'GRIP_STRENGHT': dataset["GRIP_STRENGHT"],
        'MUSCLE_MASS': dataset["MUSCLE_MASS"]
    }, orient="index").T
    '''
    print("Dimensioni di tutte le misurazioni: {}".format(dataset.shape))
    # print(dataset.shape)
    new_model = IsolationForest(contamination=0.1)  # random_state=0, contamination=0.01
    new_model.fit(dataset)
    database.db_insert_model(DB, new_model, ID)
    print("[RE-TRAIN MODEL] - End!")

    return '{ "ML success": 1 }'


@app.route('/api/senddata', methods=['GET', 'POST'])
def data_receiver():
    """
    paziente --> codice_fiscale (o ID)
    misurazioni --> (no) FM, FFM, acc_x, acc_y, acc_z, muscle_strenght (mai, al massimo sarà lo smartphone a combinarli)
                --> (si) GAIT_SPEED, GRIP_STRENGHT, MUSCLE_MASS
    """

    ''' DATA RECEIVED BY REST POST '''
    data = request.get_json()
    data['Anomaly'] = -1    # Not checked
    print('Data received: {}'.format(data))

    DB = 'SarcopeniaDB'
    MEASUREMENT = 'measurement'
    MEASUREMENT_TEMP = 'measurement_temp'
    # ID = data["ID"]
    # GAIT_SPEED = data["GAIT_SPEED"]
    # GRIP_STRENGHT = data["GRIP_STRENGHT"]
    # MUSCLE_MASS = data["MUSCLE_MASS"]
    # X = [GAIT_SPEED, GRIP_STRENGHT, MUSCLE_MASS]

    ''' inserire nuovo dato nel DB '''
    database.db_insert_one(DB, MEASUREMENT, data)
    #database.db_insert_one(DB, MEASUREMENT_TEMP, data)

    return '{ "data success": 1 }'


if __name__ == '__main__':
    app.run()
