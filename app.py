from flask import Flask, request
from mqtt_r import connect_mqtt, subscribe
from database import connect_db, insert_one


app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/senddata', methods=['GET', 'POST'])
def data_receiver():

    """
    # JSON FORMAT [START] ----------------------------------------------------------------------------------------
    paziente --> codice_fiscale
    misurazioni --> FM, FFM, acc_x, acc_y, acc_z, muscle_strenght
    # JSON FORMAT [END] ----------------------------------------------------------------------------------------
    """

    """
    # DEBUG [START] : Insted of receive data from a POST request, use MQTT (jumping App-Android step) ------------
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()
    # DEBUG [END] ------------------------------------------------------------------------------------------------
    """

    # NORMAL BEHAVIOR [START] : DATA COLLECTION BY POST METHOD ----------------------------------------------------
    data = request.get_json()
    print('Data received: {}'.format(data))
    # NORMAL BEHAVIOR [END]: DATA COLLECTION BY POST METHOD --------------------------------------------------------

    # SAVE DATA INTO DB [START] -----------------------------------------------------------------------------------
    MEASUREMENT = 'measurement'
    insert_one(MEASUREMENT, data)
    # SAVE DATA INTO DB [END] -----------------------------------------------------------------------------------
    return '{ "success": 1 }'


if __name__ == '__main__':
    app.run()

