import os
import subprocess
import pandas as pd
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from joblib import Parallel, delayed

from utils.mongo import db
import utils.nlp


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

    def firstpos(self):
        """
        Create terms.positions
        """"
        terms = [self.model.id2word[i] for i in range(len(model.id2word.keys()))]

        # load truncated documents from the database
        cursor = db['docs.truncated'].find({}, {'lemma': 1})

        docs = []
        for term in terms:
            tmp = {'_id': term, 'firstPositions': []}
            for doc in cursor:
                pos = doc['lemma'].split('\t').index(term)
                tmp.firstPositions.append({'qid': doc['_id'], 'position': pos})
            docs.append(tmp)

        db['terms.positions'].remove({})
        db['terms.positions'].insert_many(docs)


    def doctopics(self):
        """
        Create docs.topics
        """"
        doctopics = model.load_document_topics()

        docs = []
        for qid, row in enumerate(doc_topics):
            topics = []
            for topic, probability in row:
                topics.append({'topicId': topic, 'probability': probability})
            docs.append({'_id': qid, 'topics': topics})

        db['docs.topics'].remove({})
        db['docs.topics'].insert_many(docs)

    def topicterms(self):
        """"
        Create terms.topics
        """
        phi = self.model.load_word_topics()

        # normalize and smooth document topics
        beta = 0.01 # mallet default
        phi = phi + beta
        termtopics = np.apply_along_axis(lambda x: x / x.sum(), 0, phi)
        topicterms = np.transpose(termtopics)

        docs = []
        for i in range(topicterms.shape[1]):
            term = self.mode.id2word[i]
            topics = [{'topicId': j+1, 'probability': p} for j, p in
                      enumerate(topicterms[term])]
            docs.append({'_id': term, 'topics': topics})

        db['terms.topics'].remove({})
        db['terms.topics'].insert_many(docs)

    def topics(self):
        """
        Create topics
        """"
        # TODO everything (see topics_to_mongo.R)
        pass

    def update(self):
        self.train()
        self.load_model()
        self.firstpos()
        self.doctopics()
        self.topicterms()
        self.topics()
