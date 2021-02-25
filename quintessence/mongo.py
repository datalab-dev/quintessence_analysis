from joblib import delayed
from joblib import Parallel
import numpy as np
import pandas as pd
from pymongo import MongoClient

from quintessence.nlp import normalize_text
from quintessence.parse_topicmodel import create_doc_topics
from quintessence.parse_topicmodel import create_topic_terms
from quintessence.parse_topicmodel import create_topic_topterms
from quintessence.parse_topicmodel import create_topics
from quintessence.parse_embed import get_all_vocab
from quintessence.parse_embed import create_nearest_neighbors
from quintessence.parse_embed import create_similarity_over_time
from quintessence.parse_embed import create_subsets
from quintessence.parse_embed import create_terms
from quintessence.parse_embed import get_vocab
from quintessence.wordcounts import create_corpus_frequencies
from quintessence.wordcounts import create_doc_frequencies
from quintessence.wordcounts import create_term_frequencies

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

    def write_topic_model_data(self, corpus, lda):
        """
        Given a trained TopicModel class, create and write all the necessary 
        data to the mongo database.
    
        tables that are overwritten (deleted then created)
            - doc.topics
            - topic.terms
            - topics
        """

        # topic.topterms
        print("topic top terms")
        self.db['topics.topterms'].remove({})
        self.db['topics.topterms'].insert_many(
                create_topic_topterms(lda.topicterms))

        # doc.topics
        print("doc topics")
        self.db['docs.topics'].remove({})
        self.db['docs.topics'].insert_many(create_doc_topics(lda.doctopics))

        # topic.terms
        print("topic terms")
        self.db['topics.terms'].remove({})
        self.db['topics.terms'].insert_many(
                create_topic_terms(lda.topicterms))

        # topics
        print("topics")
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

    def write_frequency_data(self):
        workers = self.workers

        print("preprocess")
        corpus = self.get_embeddings_data() # std 
        corpus["raw_word_count"] = corpus["docs"].apply(lambda x: len(x.split()))
        corpus["docs"] = Parallel(n_jobs = workers)(delayed(
            normalize_text)(d) for d in corpus["docs"])
        corpus["word_count"] = corpus["docs"].apply(len)

        print("doc frequencies")
        self.db["frequencies.docs"].remove({})
        self.db["frequencies.docs"].insert_many(
                create_doc_frequencies(corpus))

        print("corpus frequencies")
        self.db["frequencies.corpus"].remove({})
        self.db["frequencies.corpus"].insert(
                create_corpus_frequencies(corpus))

        print("frequencies terms")
        self.db["frequencies.terms"].remove({})
        self.db["frequencies.terms"].insert_many(
                create_term_frequencies(corpus))
