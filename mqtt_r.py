from paho.mqtt import client as mqtt_client
from database import connect_db, insert_one
import json

broker = 'mqtt.eclipseprojects.io'
port = 1883
topic_bia = 'unisalento/sarcopenia/data/bia'
topic_acceleration = 'unisalento/sarcopenia/data/acceleration'
topic_musclestrenght = 'unisalento/sarcopenia/data/muscle_strenght'
client_id = 'mqtt_unisalento_sarcopenia_r'
# username = 'emqx'
# password = 'public'


def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        insert_one('measurement', json.loads(msg.payload.decode()))

    client.subscribe(topic_bia)
    client.on_message = on_message

'''
def mqtt_run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()
'''


