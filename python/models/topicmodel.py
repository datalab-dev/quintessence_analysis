import os
import subprocess
import pandas as pd

from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from joblib import Parallel, delayed

class TopicModel:
    def __init__(self, model_path, mallet_path=None, num_topics=None):
        self.model_path = model_path # fname for mallet model
        self.mallet_path = mallet_path # ex: '~/mallet-2.0.8/bin/mallet'
        self.num_topics = num_topics
        self.model = None # LdaMallet obj

    def train(self):
        """
        Train topic model using mallet.
        """
        # load the corpus from mongo
        cursor = db.docs.find(projection={'lemma': 1}).sort('_id', 1)
        docs = cursor[:]

        # split and clean
        tokenized = Parallel(n_jobs=80)(
            delayed(nlp.tokenize)(doc['lemma'], True) for doc in docs)

        dictionary = Dictionary(tokenized)
        dictionary.filter_extremes(no_below=200, no_above=0.2) # prune
        corpus = [dictionary.doc2bow(doc) for doc in tokenized]

        # train model
        model = LdaMallet(self.mallet_path, corpus=corpus,
                          num_topics=self.num_topics, id2word=dictionary)

        # save models
        self.model = model
        model.save(self.model_path)

    def load_model(self):
        """
        Load a previously saved model.
        """
        self.model = LdaMallet.load(mallet_path)
