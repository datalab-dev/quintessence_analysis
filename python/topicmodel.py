import os
import subprocess
import pandas as pd
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary

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
            delayed(nlp.tokenize)(doc['lemma']) for doc in docs)

        dictionary = Dictionary(tokenized)
        dictionary.filter_extremes(no_below=200, no_above=0.2) # prune
        corpus = [dictionary.doc2bow(doc) for doc in tokenized]

        # train model
        model = LdaMallet(self.mallet_path, corpus=corpus,
                          num_topics=self.num_topics)

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
        # TODO load a list of terms from somewhere

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
        # TODO load a dataframe from somewhere (change below line based on
        # how mallet outputs are parsed)
        # using gensim model.load_document_topics()
        theta = pd.read_csv(self.model_dir + 'doctopics.dat', sep='\t',
                            header=None)
        fnames = theta[1]
        theta.head()
        del theta[0] # row numbers
        del theta[1] # filenames

        # normalize and smooth document topics
        alpha = 5 / theta.shape[1] # mallet default is 5/K
        theta = theta + alpha # smooth
        doctopics = theta.div(theta.sum(axis=0), axis=1) # normalize columns

        docs = []
        for i, fname in enumerate(fnames):
            fileid = os.path.splitext(os.path.basename(fname))[0]
            res = db['docs.metadata'].find_one({'fileId': fileid}, {'_id': 1})
            qid = res['_id']

            topics = [{'topicId': j+1, 'probability': p} for j, p in
                      enumerate(doctopics.iloc[i])]
            docs.append({'_id': qid, 'topics': topics})

        db['docs.topics'].remove({})
        db['docs.topics'].insert_many(docs)

    def topicterms(self):
        """"
        Create terms.topics
        """
        # TODO load a dataframe from somewhere
        # model.get_topics() [num_topics x vocabulary_size]

        # normalize and smooth topic terms
        beta = 0.01 # mallet default
        phi = phi + beta
        topicterms = phi.div(phi.sum(axis=1), axis=0) # normalize rows

        docs = []
        for term in topicterms.columns:
            topics = [{'topicId': j+1, 'probability': p} for j, p in
                      enumerate(topicterms[term])]
            docs.append({'_id': term, 'topics': topics})

        db['terms.topics'].remove({})
        db['terms.topics'].insert_many(docs)

    def topics(self):
        """
        Create topics
        """"
        pass

    def update(self):
        self.train()
        self.load_model()
        self.firstpos()
        self.doctopics()
        self.topicterms()
        self.topics()
