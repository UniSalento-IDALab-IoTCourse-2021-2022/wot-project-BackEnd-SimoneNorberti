from flask import Flask, request
from mqtt_r import connect_mqtt, subscribe


app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/senddata', methods=['GET', 'POST'])
def datareceiver():
    """
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()
    """
    data = request.get_json()
    print('Data received:{}'.format(data))
    # TODO inserire tali dati nel DB

    return 'Provo a ricevere i dati MQTT'


if __name__ == '__main__':
    app.run()

