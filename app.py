import pandas as pd
from flask import Flask, request

import database
import json
from sklearn.ensemble import IsolationForest

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


# TODO inserire API per "Ferma allenamento" in modo da avviare l'algoritmo di ML


@app.route('/senddata', methods=['GET', 'POST'])
def data_receiver():
    """
    paziente --> codice_fiscale (o ID)
    misurazioni --> (no) FM, FFM, acc_x, acc_y, acc_z, muscle_strenght
                --> (si) GAIT_SPEED, GRIP_STRENGHT, MUSCLE_MASS
    """

    # DATA RECEIVED BY POST
    data = request.get_json()   # json_
    print('Data received: {}'.format(data))
    json_dict = json.dumps(data)
    # data = json.loads(json_dict)

    # print(type(json_data))
    # print(type(json_dict))
    print(type(data))

    DB = 'SarcopeniaDB'
    MEASUREMENT = 'measurement'
    ID = data["ID"]
    GAIT_SPEED = data["GAIT_SPEED"]
    GRIP_STRENGHT = data["GRIP_STRENGHT"]
    MUSCLE_MASS = data["MUSCLE_MASS"]
    X = [GAIT_SPEED, GRIP_STRENGHT, MUSCLE_MASS]

    # 1: fare anomaly detection - controllo inizizzazione nuovo utente
    model = database.db_get_model(DB, ID)
    if model is None:
        if database.db_count(DB, MEASUREMENT, ID) > 10:
            dataset = database.db_get_all(DB, MEASUREMENT, ID)
            print(dataset)
            new_model = IsolationForest(random_state=0)
            new_model.fit(dataset)
            database.db_insert_model(DB, new_model, ID)
            database.db_insert_one(DB, MEASUREMENT, data)
            return '{ "success": 2 }'
        else:
            database.db_insert_one(DB, MEASUREMENT, data)
            return '{ "success": 3 }'

    # 2: inserire nuovo dato nel DB
    database.db_insert_one(DB, MEASUREMENT, data)

    # 3: fare l'anomaly detection
    data_to_predict = pd.DataFrame.from_dict({
        'GAIT_SPEED': data["GAIT_SPEED"],
        'GRIP_STRENGHT': data["GRIP_STRENGHT"],
        'MUSCLE_MASS': data["MUSCLE_MASS"]
    }, orient="index").T
    print(data_to_predict)
    pred = model.predict(data_to_predict)
    if pred == 1:
        print("Anomalia Rilevata - model.predict(data): {}".format(pred))
        # TODO AVVISARE IL MEDICO
    elif pred == -1:
        print("No anomalia - model.predict(data): {}".format(pred))


    # 4: eliminare il vecchio modello
    database.db_delete_model(DB, ID)

    # 5: train new model
    dataset = database.db_get_all(DB, MEASUREMENT, ID)
    new_model = IsolationForest(random_state=0)
    new_model.fit(dataset)
    database.db_insert_model(DB, new_model, ID)

    return '{ "success": 1 }'


if __name__ == '__main__':
    app.run()

"""
# DEBUG [START] : Insted of receive data from a POST request, use MQTT (jumping App-Android step) 
client = connect_mqtt()
subscribe(client)
client.loop_forever()
"""
