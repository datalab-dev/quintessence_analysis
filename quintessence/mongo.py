import json

import pandas as pd
import numpy as np
from pymongo import MongoClient
from gensim.models.wrappers import LdaMallet

from quintessence.topicmodel import TopicModel
from quintessence.nlp import compute_proportions
from quintessence.nlp import compute_coordinates

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
        """ returns full list from docs.meta """
        meta = list(self.db["docs.meta"].find({}))
        return meta

    def get_topic_model_data(self):
        """ return ids, and strings of lemmatized documents """
        res = list(self.db["docs.lemma"].find({}))
        docs = [" ".join(r["lemma"].split('\t')) for r in res]
        ids = [r["_id"] for r in res]
        return (ids, docs)

    def write_topic_model_data(self, lda):
        """
        Given a trained TopicModel class, create and write all the necessary 
        data to the mongo database.
    
        tables that are overwritten (deleted then created)
            - doc.topics
            - topic.terms
            - topics
        """

        topicterms = lda.topicterms
        topicterms = np.apply_along_axis(lambda x: x / x.sum(), 1, topicterms) # normalize and smooth document topics
        termstopics = topicterms.T
        doctopics = lda.doctopics.todense().A

        # Create docs.topics
        # _id: 0,
        # topics: [
        #    {"topicId": 0, "probability": 0.05},
        #    {"topicId": 1, "probability": 0.08}, ...
        #    ]
        docs = []
        for qid, row in enumerate(doctopics):
            topics = []
            for topic, probability in enumerate(row):
                topics.append({'topicId': topic, 'probability': probability})
            docs.append({'_id': qid, 'topics': topics})

        self.db['docs.topics'].remove({})
        self.db['docs.topics'].insert_many(docs)

        # topic.terms
        # topicId: 0,
        # terms: [
        #    {"term": "abate", "probability": 0.01}, ...
        # ]
        docs = []
        for topicid, row in enumerate(topicterms):
            terms = []
            for termindex, probability in enumerate(row):
                term = lda.dictionary.id2token[termindex]
                terms.append({'term': term, 'probability': probability})
            docs.append({'topicId': topicid, 'terms': terms})

        self.db['topics.terms'].remove({})
        self.db['topics.terms'].insert_many(docs)

        # Create topics
        # proportion: 0.0294,
        # x: -0.13,
        # y: 0.115,
        # authors: [...],
        # locations: [...],
        # keywords: [...],
        # publishers: [...],
        # topDocs: [1, 5, 345, 657, 34503]

         proportions = compute_proportions(doc_topics, doc_lens)
         coordinates = compute_coordinates(topic_terms)
         # topdocs = compute_top_docs(doc_topics)
         # authors = compute_top(doc_topics, "authors")
         # keywords = compute_top(doc_topics, "keywords")
         # locations = compute_top(doc_topics, "locations")
         # publishers = compute_top(doc_topics, "publishers")
         # meta is calculated as such:
         # filter docs based on meta
         # get mean of nonzeros of topic proportion for each subset for each topic
#         meta = pd.DataFrame.from_records(self.get_metadata())




