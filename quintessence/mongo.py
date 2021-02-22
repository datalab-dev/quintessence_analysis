from collections import Counter
import json

from joblib import delayed
from joblib import Parallel
from gensim.corpora import Dictionary
import numpy as np
import pandas as pd
from pymongo import MongoClient

from quintessence.nlp import normalize_text
from quintessence.parse_topicmodel import create_doc_topics
from quintessence.parse_topicmodel import create_topic_terms
from quintessence.parse_topicmodel import create_topics
from quintessence.parse_embed import get_all_vocab
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

    def write_word_frequency_data(self):
        """ ONLY RUN AFTER 'terms' collection exists! ie. after embed """ 
        corpus = self.get_embeddings_data() #std

        # normalize text
        docs = corpus["docs"]
        normalized = [normalize_text(d) for d in docs]
        corpus["docs"] = normalized

        # add decades and word_count column to metadata
        corpus["decade"] = corpus["Date"].apply(lambda x: x[0:3] + '0')
        corpus["word_count"] = corpus["docs"].apply(len)

        # get ndocs / year, ndocs / decade
        ndocs_per_year = corpus["Date"].value_counts()
        ndocs_per_decade = corpus["decade"].value_counts()

        # get nterms / year, nterms / decade
        ntokens_per_year = corpus.groupby["Date"].sum("word_count")
        ntokens_per_decade = corpus.groupby["decade"].sum("word_count")

        # for each term in terms table, get year:count ... decade:count 
        # collapse docs on years -> create dtm where rows are years
        # collapse docs on decades -> create dtm where rows are decades
        dictionary = Dictionary(normalized)
        dictionary.doc2bow(doc)
        dtm = corpus2csc(dtm).T
        dtm = dtm.toarray()
        dtm = pd.DataFrame(data = dtm, index=years)


        # add terms.frequencies table

    def write_topic_model_data(self, corpus, lda):
        """
        Given a trained TopicModel class, create and write all the necessary 
        data to the mongo database.
    
        tables that are overwritten (deleted then created)
            - doc.topics
            - topic.terms
            - topics
        """

        # doc.topics
        self.db['docs.topics'].remove({})
        self.db['docs.topics'].insert_many(create_doc_topics(lda.doctopics))

        # topic.terms
        self.db['topics.terms'].remove({})
        self.db['topics.terms'].insert_many(
                create_topic_terms(lda.topicterms))

        # topics
        self.db['topics'].remove({})
        self.db['topics'].insert_many(create_topics(corpus,
            lda.doctopics, 
            lda.dtm, lda.topicterms))

    def write_embeddings_data(self, embeddings):
        print("get vocab")
        vocab = get_all_vocab(embeddings.model, embeddings.subsets, embeddings.decades)

        # terms
        print("terms")
        self.db['terms'].remove({})
        self.db['terms'].insert_many(create_terms(embeddings.model, vocab))

        # terms.subsets
        print("subsets")
        self.db['terms.subsets'].remove({})
        self.db['terms.subsets'].insert_many(create_subsets(embeddings.subsets))

        # nearest neighbors
        collections = self.db.list_collection_names()

        print("nearest neighbors subsets")
        ## subsets
        subset_collections = [c for c in collections if "terms.subset." in c]
        for sc in subset_collections:
            self.db[sc].remove({})
        for s in embeddings.subsets:
            # terms.subset.name
            self.db["terms.subset." + s[0]].insert_many(
                create_nearest_neighbors(s, vocab))

        ## decades
        print("nearest neighbors decades")
        decades_collections = [c for c in collections if "terms.decade." in c]
        for dc in decades_collections:
            self.db[dc].remove({})
        for d in embeddings.decades:
            # terms.decade.name
            self.db["terms.decade." + d[0]].insert_many(
                create_nearest_neighbors(d, vocab))

        # terms.timeseries
        print("time series")
        self.db['terms.timeseries'].remove({})
        self.db['terms.timeseries'].insert_many(
            create_similarity_over_time(embeddings.decades, vocab))
