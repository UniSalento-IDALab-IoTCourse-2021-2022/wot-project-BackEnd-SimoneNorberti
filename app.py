from flask import Flask, request
from mqtt_r import connect_mqtt, subscribe
from database import db_connect, db_insert_one
from machineLearning import train_send_model

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


# TODO inserire API per "Ferma allenamento" in modo da avviare l'algoritmo di ML


@app.route('/senddata', methods=['GET', 'POST'])
def data_receiver():
    """
    paziente --> codice_fiscale
    misurazioni --> FM, FFM, acc_x, acc_y, acc_z, muscle_strenght
    """

    # DATA COLLECTION BY POST METHOD
    data = request.get_json()
    print('Data received: {}'.format(data))

    # SAVE DATA INTO DB
    DB = 'SarcopeniaDB'
    MEASUREMENT = 'measurement'
    db_insert_one(db_connect(DB), MEASUREMENT, data)

    return '{ "success": 1 }'


@app.route('/receivemodel', methods=['GET'])
def send_model():
    return train_send_model()


if __name__ == '__main__':
    app.run()

"""
# DEBUG [START] : Insted of receive data from a POST request, use MQTT (jumping App-Android step) 
client = connect_mqtt()
subscribe(client)
client.loop_forever()
"""
