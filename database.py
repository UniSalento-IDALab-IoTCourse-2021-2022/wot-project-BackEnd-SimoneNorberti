import sys
import pymongo as pymongo
import datetime
import pickle
import pandas as pd


# TODO inserire anche timestamp
# TODO il giorno dell'esame ricordarsi che serve un IP autorizzato


'''
def db_connect(db):

    return db
'''

'''
def db_update_one(db, collection, ID, count):
    client = pymongo.MongoClient("mongodb+srv://snorb:7OGFhqrLw8rTfCaL@clustersarcopenia0.gvzw6w6.mongodb.net/"
                                 "?retryWrites=true&w=majority")
    db = client['' + db + '']
    collection = db['' + collection + '']
    filter_query = {'ID': ID}
    update_query = {'$set': {'count': count}}
    return collection.update_one(filter_query, update_query)
'''


def db_delete_all(db, collection, ID):
    print("         --> chiamata la funzione db_delete_all()")
    client = pymongo.MongoClient("mongodb+srv://snorb:7OGFhqrLw8rTfCaL@clustersarcopenia0.gvzw6w6.mongodb.net/"
                                 "?retryWrites=true&w=majority")
    db = client['' + db + '']
    collection = db['' + collection + '']

    # Eliminazione di tutti i documenti nella collection
    query = {"ID": ID}
    result = collection.delete_many(query)

    # Restituzione del numero di documenti eliminati
    return result.deleted_count


def db_insert_one(db, collection, data):
    client = pymongo.MongoClient("mongodb+srv://snorb:7OGFhqrLw8rTfCaL@clustersarcopenia0.gvzw6w6.mongodb.net/"
                                 "?retryWrites=true&w=majority")
    db = client['' + db + '']
    collection = db['' + collection + '']
    try:
        id_insert = collection.insert_one(data).inserted_id
    except:
        print("[DB ERROR] insert_one, INFO:", sys.exc_info()[0])
        return
    print("[DB OK] insert_one - id: {}".format(id_insert))
    return


def db_insert_model(db, model, ID):
    client = pymongo.MongoClient("mongodb+srv://snorb:7OGFhqrLw8rTfCaL@clustersarcopenia0.gvzw6w6.mongodb.net/"
                                 "?retryWrites=true&w=majority")
    db = client['' + db + '']

    # Creazione della raccolta "models" se non esiste già
    if "models" not in db.list_collection_names():
        db.create_collection("models")

    # Serializzazione del modello in formato pickle
    model_pickled = pickle.dumps(model)

    # Inserimento del modello nel database
    model_doc = {"name": ID, "model": model_pickled}
    models_coll = db["models"]
    models_coll.insert_one(model_doc)


def db_get_model(db, ID):
    client = pymongo.MongoClient("mongodb+srv://snorb:7OGFhqrLw8rTfCaL@clustersarcopenia0.gvzw6w6.mongodb.net/"
                                 "?retryWrites=true&w=majority")
    db = client['' + db + '']
    # Recupero del modello dal database
    models_coll = db["models"]
    model_doc = models_coll.find_one({"name": ID})
    if model_doc is None:
        return
    model_pickled = model_doc["model"]
    model = pickle.loads(model_pickled)
    return model


def db_delete_model(db, ID):
    client = pymongo.MongoClient("mongodb+srv://snorb:7OGFhqrLw8rTfCaL@clustersarcopenia0.gvzw6w6.mongodb.net/"
                                 "?retryWrites=true&w=majority")
    db = client['' + db + '']
    # Cancellazione del modello dal database
    models_coll = db["models"]
    models_coll.delete_one({"name": ID})


def db_get_all(db, collection, ID):
    client = pymongo.MongoClient("mongodb+srv://snorb:7OGFhqrLw8rTfCaL@clustersarcopenia0.gvzw6w6.mongodb.net/"
                                 "?retryWrites=true&w=majority")
    db = client['' + db + '']
    collection = db['' + collection + '']
    # Query per recuperare tutte le occorrenze con un attributo specifico
    query = {"ID": ID}
    results = collection.find(query)
    df = pd.DataFrame(list(results))
    # print('db_get_all(): df:{}'.format(df))
    # print('db_get_all(): df.iloc[:,[1, 2, 3]]:{}'.format(df.iloc[:, [1, 2, 3]]))
    return df.iloc[:, [1, 2, 3]]


def db_count(db, collection, ID):
    client = pymongo.MongoClient("mongodb+srv://snorb:7OGFhqrLw8rTfCaL@clustersarcopenia0.gvzw6w6.mongodb.net/"
                                 "?retryWrites=true&w=majority")
    db = client['' + db + '']
    collection = db['' + collection + '']
    # Query per contare il numero di documenti con un certo attributo
    query = {"ID": ID}
    count = collection.count_documents(query)
    print('db_count(): {}'.format(count))
    return count
