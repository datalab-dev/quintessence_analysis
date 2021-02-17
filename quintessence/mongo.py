import json

import pandas as pd
import numpy as np
from pymongo import MongoClient

from quintessence.parse_topicmodel import create_doc_topics
from quintessence.parse_topicmodel import create_topic_terms
from quintessence.parse_topicmodel import create_topics
from quintessence.parse_embed import create_nearest_neighbors
from quintessence.parse_embed import create_similarity_over_time
from quintessence.parse_embed import create_subsets
from quintessence.parse_embed import create_terms
from quintessence.parse_embed import get_vocab

class Mongo:
    db = None

    def __init__(self, credentials):
        """ create a connection to the mongo database """
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

    def write_embeddings_data(self, embeddings):
        vocab = get_vocab(embeddings.model)

        # terms
        self.db['terms'].remove({})
        self.db['terms'].insert_many(create_terms(embeddings.model, vocab))

        # terms.subsets
        self.db['terms.subsets'].remove({})
        self.db['terms.subsets'].insert_many(create_subsets(embeddings.subsets))

        # nearest neighbors
        collections = self.db.list_collection_names()

        ## subsets
        subset_collections = [c for c in collections if "terms.subset." in c]
        for sc in subset_collections:
            self.db[sc].remove({})
        for s in embeddings.subsets:
            # terms.subset.name
            self.db["terms.subset." + s[0]].insert_many(
                create_nearest_neighbors(s, vocab))

        ## decades
        decades_collections = [c for c in collections if "terms.decade." in c]
        for dc in decades_collections:
            self.db[dc].remove({})
        for d in embeddings.decades:
            # terms.decade.name
            self.db["terms.decade." + d[0]].insert_many(
                create_nearest_neighbors(d, vocab))

        # terms.timeseries
        self.db['terms.timeseries'].remove({})
        self.db['terms.timeseries'].insert_many(
            create_similarity_over_time(embeddings.decades, vocab))
