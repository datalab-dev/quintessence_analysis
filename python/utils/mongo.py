import json
from pymongo import MongoClient


def get_mongo_db(credentials_path):
    # construct mongo url
    with open(credentials_path, 'r') as f:
        credentials = json.load(f)
        url = f"mongodb://{credentials['server']}:{credentials['port']}"

    # try to form client connection
    try:
        client = MongoClient(url)
    except pymongo.errors.ConnectionFailure:
         print(f"Failed to connect to {url}")

    return client[credentials['dbname']]


db = get_mongo_db('../../mongo_credentials.json')
