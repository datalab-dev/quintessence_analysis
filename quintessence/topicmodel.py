import os

from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
import numpy as np

class TopicModel:
    def __init__(self, model_odir, mallet_path=None, num_topics=None):
        self.model_odir = model_odir # dirname for mallet model, temp files, and dictionary ...
        self.mallet_path = mallet_path # ex: '~/mallet-2.0.8/bin/mallet'
        self.num_topics = num_topics
        self.model = None # LdaMallet obj
        self.corpus = None
        self.dictionary = None
        self.fnames = None
        self.vocab = None
        self.word_counts = None
        self.doc_lens = None
        self.topicterms = None
        self.doctopics = None

    def train(self, docs, fnames):
        """
        Train topic model using mallet.

        Expects docs to be a list of normalized strings where each string is a unique document
        fnames should map to docs
        """
        docs = [doc.split() for doc in docs]
        self.dictionary = Dictionary(docs)
        self.dictionary.filter_extremes(no_below=int(0.01 * len(docs)), no_above=0.8) # prune
        self.corpus = [self.dictionary.doc2bow(doc) for doc in docs]

        # train model
        model = LdaMallet(self.mallet_path,
                corpus=self.corpus, prefix = self.model_odir,
                          num_topics=self.num_topics, id2word=self.dictionary)

        tdm = corpus2csc(self.corpus)
        self.topicterms = model.get_topics()
        self.doctopics = model.load_document_topics()
        self.doc_lens = np.asarray(tdm.sum(axis=0))
        self.fnames = fnames
        self.vocab = [t for t in self.dictionary.itervalues()]
        self.word_counts = np.asarray(tdm.sum(axis=1))

        # save model, dictionary, corpus
        self.model = model
        model.save(self.model_odir) 

    def load_model(self):
        """
        Load a previously saved model and the other files that should be in the odir
        """
        self.model = LdaMallet.load(self.model_path)
