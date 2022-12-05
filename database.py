import sys
import pymongo as pymongo
import datetime
# TODO inserire anche timestamp

"""
collection: paziente
    codice_fiscale: Primary
    nome: 
    cognome:
    data_nascita:
    luogo_nascita:

collection: misurazioni
    FM:
    FFM:
    acc_x:
    acc_y:
    acc_z:
    muscle_strenght:
"""


def connect_db(db):
    client = pymongo.MongoClient("mongodb+srv://snorb:K3Bjj4wD9VIzlg1E@clustersarcopenia0.gvzw6w6.mongodb.net/"
                                     "?retryWrites=true&w=majority")
    db = client['' + db + '']
    return db


def insert_one(collection, data):
    # Try to connect to DB
    try:
        db = connect_db('SarcopeniaDB')
    except:
        print("[ERROR] Connection to DB failed!, INFO:", sys.exc_info()[0])
        return
    print("[OK] Connection to DB successful!")

    collection = db['' + collection + '']
    # Try to do an "insert_one" operation
    try:
        prova = collection.insert_one(data).inserted_id
    except:
        print("[ERROR] insert_one, INFO:", sys.exc_info()[0])
        return
    print("[OK] insert_one - id: {}".format(prova))
    return
