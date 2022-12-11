import sys
import pymongo as pymongo
import datetime

# TODO inserire anche timestamp
# TODO il giorno dell'esame ricordarsi che serve un IP autorizzato
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


def db_connect(db):
    client = pymongo.MongoClient("mongodb+srv://snorb:K3Bjj4wD9VIzlg1E@clustersarcopenia0.gvzw6w6.mongodb.net/"
                                 "?retryWrites=true&w=majority")
    db = client['' + db + '']
    return db


def db_insert_one(db, collection, data):
    # print("[ERROR] Connection to DB failed!, INFO:", sys.exc_info()[0])
    # print("[OK] Connection to DB successful!")
    collection = db['' + collection + '']

    try:
        id_insert = collection.insert_one(data).inserted_id
    except:
        print("[DB ERROR] insert_one, INFO:", sys.exc_info()[0])
        return
    print("[DB OK] insert_one - id: {}".format(id_insert))
    return
