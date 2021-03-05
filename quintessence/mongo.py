import pandas as pd
from pymongo import MongoClient

from quintessence.parse_embed import create_embeddings_datamodel
from quintessence.parse_topicmodel import create_topicmodel_datamodel
from quintessence.wordcounts import create_frequencies_datamodel

class Mongo:
    db = None
    workers = 4

    def __init__(self, credentials, workers=4):
        """ create a connection to the mongo database """
        self.workers = workers
        try:
            client = MongoClient(
                    f"mongodb://{credentials['host']}:{credentials['port']}")
            self.db = client[credentials['database']]
        except pymongo.errors.ConnectionFailure:
            print(f"Failed to connect to {url}")


    def get_metadata(self):
        """ returns dataframe version of metadata from database """
        meta = pd.DataFrame.from_records(list(self.db["docs.meta"].find({})))
        return meta.set_index("_id")

    def get_embeddings_data(self):
        """ returns pandas series of std data from db """

        res = list(self.db["docs.std"].find({}))
        docs = [" ".join(r["std"].split('\t')) for r in res]
        ids = [r["_id"] for r in res]

        meta = self.get_metadata()
        df = pd.DataFrame( {'_id': ids, 'docs': docs}).dropna()
        return df.set_index("_id").join(meta)

    def get_topic_model_data(self):
        """  pandas series of lemma data from db """
        res = list(self.db["docs.lemma"].find({}))
        docs = [" ".join(r["lemma"].split('\t')) for r in res]
        ids = [r["_id"] for r in res]

        meta = self.get_metadata()
        df = pd.DataFrame( {'_id': ids, 'docs': docs}).dropna()
        return df.set_index("_id").join(meta)

    def write_topic_model_data(self, lda):
        """
        Given a trained TopicModel class, create and write all the necessary 
        data to the mongo database.
        """
        collections = create_topicmodel_datamodel(lda.doctopics, 
                lda.topicterms, lda.meta, lda.dtm)

        for collection_name, documents  in collections.items():
            self.db[collection_name].remove({})
            self.db[collection_name].insert_many(documents)

    def write_embeddings_data(self, embeddings):
        collections = create_embeddings_datamodel(embeddings.full,
                embeddings.subsets, embeddings.decades)

        for collection_name, documents  in collections.items():
            self.db[collection_name].remove({})
            self.db[collection_name].insert_many(documents)

    def write_frequency_data(self):
        collections = create_frequencies_datamodel(
                corpus=self.get_embeddings_data(),
                workers =  self.workers)

        for collection_name, documents  in collections.items():
            self.db[collection_name].remove({})
            self.db[collection_name].insert_many(documents)
