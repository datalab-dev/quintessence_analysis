import pandas as pd

from utils.mongo import Mongo

mongo = Mongo("../mongo_credentials.json")
meta = pd.DataFrame.from_records(mongo.get_metadata())

