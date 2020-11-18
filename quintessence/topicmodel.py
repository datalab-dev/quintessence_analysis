import os

from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
grom gensim.matutils import corpus2csc

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
        self.corpus = [dictionary.doc2bow(doc) for doc in docs]

        # train model
        model = LdaMallet(self.model_odir + "mallet.model", 
                corpus=corpus, prefix = model_odir,
                          num_topics=self.num_topics, id2word=dictionary)

        self.topicterms = model.get_topics()
        self.doctopics = 
        self.doc_lens = 
        self.fnames = fnames
        self.vocab = 
        self.word_counts = 

        # save model, dictionary, corpus
        self.model = model
        model.save(self.model_path)

    def load_model(self):
        """
        Load a previously saved model and the other files that should be in the odir
        """
        self.model = LdaMallet.load(self.model_path)
