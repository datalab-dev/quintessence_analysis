import json

import pandas as pd
import numpy as np
from pymongo import MongoClient

from quintessence.parse_lda import create_doc_topics
from quintessence.parse_lda import create_topic_terms
from quintessence.parse_lda import create_topics

class Mongo:
    def __init__(self, credentials_path):
        """ create a connection to the mongo database """
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
            url = f"mongodb://{credentials['host']}:{credentials['port']}"

        # try to form client connection
        try:
            client = MongoClient(url)
        except pymongo.errors.ConnectionFailure:
            print(f"Failed to connect to {url}")

        self.db = client[credentials['database']]

    def get_metadata(self):
        """ returns dataframe version of metadata from database """
        meta = pd.DataFrame.from_records(list(self.db["docs.meta"].find({})))
        return meta

    def get_embeddings_data(self):
        """ returns pandas series of std data from db """
        res = list(self.db["docs.std"].find({}))
        docs = [" ".join(r["std"].split('\t')) for r in res]
        ids = [r["_id"] for r in res]
        return pd.Series(docs, index=ids)

    def write_embeddings_data(self):
        """  """
        # TODO:
        pass

    def get_topic_model_data(self):
        """  pandas series of lemma data from db """
        res = list(self.db["docs.lemma"].find({}))
        docs = [" ".join(r["lemma"].split('\t')) for r in res]
        ids = [r["_id"] for r in res]
        return pd.Series(docs, index=ids)

    def write_topic_model_data(self, lda):
        """
        Given a trained TopicModel class, create and write all the necessary 
        data to the mongo database.
    
        tables that are overwritten (deleted then created)
            - doc.topics
            - topic.terms
            - topics
        """

        meta = pd.DataFrame.from_records(self.get_metadata())
        # doc.topics
        self.db['docs.topics'].remove({})
        self.db['docs.topics'].insert_many(create_doc_topics(lda.doctopics))

        # topic.terms
        self.db['topics.terms'].remove({})
        self.db['topics.terms'].insert_many(
                create_topic_terms(lda.topicterms, lda.dictionary))

        # topics
        self.db['topics'].remove({})
        self.db['topics'].insert_many(create_topics(meta,
            lda.doctopics, 
            lda.doc_lens, lda.topicterms))
