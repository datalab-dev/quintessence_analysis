import os
import pickle

from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
import numpy as np

class TopicModel:
    def __init__(self, model_odir, mallet_path=None, num_topics=None):
        """
        Initialize a TopicModel class, either to use for training models or to load in a model trained by this class.

        Currently uses Gensim's wrapper of LdaMallet as the lda implementation.
        Provides methods for - training a model, loading in a model.
        In addition to the model object output from gensim, this class also saves
        - information about the training (num_topics)
        - the gensim corpus object that was modeled
        - dictionary object for the corpus
        - qids of the documents
        - word_counts for the documents (at the time they are modeled, i.e after preprocessing)
        - vocab, the list of words
        - topicterms matrix
        - doctopics matrix

        """
        # dirname for mallet model, temp files, and dictionary ...
        # note if expanduser not set then it will incorrectly handle ~ 
        # when setting absolute; for example test os.path.abspath(~/Documents)
        self.model_odir = os.path.abspath(os.path.expanduser(model_odir))
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

        if not os.path.isdir(self.model_odir):
            os.mkdir(self.model_odir)

    def train(self, docs):
        """
        Train topic model using mallet.

        Docs is pandas series, index is ids, and values are normalized strings

        Saves all outputs as properties of the class instance, as well as to files in self.model_odir path
        """
        self.fnames = list(docs.index)
        docs = [doc.split() for doc in docs]
        self.dictionary = Dictionary(docs)
        self.dictionary.filter_extremes(no_below=int(0.01 * len(docs)), no_above=0.8) # prune
        self.corpus = [self.dictionary.doc2bow(doc) for doc in docs]

        # train model
        # mallet is a dummy? add / to prefix...
        self.model = LdaMallet(self.mallet_path,
                corpus=self.corpus, prefix = self.model_odir + "/",
                          num_topics=self.num_topics, id2word=self.dictionary)

        tdm = corpus2csc(self.corpus)
        self.topicterms = self.model.get_topics() + 0.01 # mallet default beta
        self.doctopics = corpus2csc([i for i in self.model.load_document_topics()]).T
        self.doc_lens = np.asarray(tdm.sum(axis=0))
        self.vocab = [t for t in self.dictionary.itervalues()]
        self.word_counts = np.asarray(tdm.sum(axis=1))

        # save model, dictionary, corpus, fnames
        self.dictionary.save_as_text(self.model_odir + "/dict.txt") 
        with open(self.model_odir + "/corpus.pickle", "wb") as cp:
            pickle.dump(self.corpus, cp)
        with open(self.model_odir + "/fnames.pickle", "wb") as fp:
            pickle.dump(self.fnames, fp)
        self.model.save(self.model_odir + "/mallet.model") 


    def load_model(self):
        """
        Load a previously saved model and the other files that should be in the odir
        updates all the class instance properties (vocab, doctopics ...)
        Does not require a mallet path
        """
        self.model = LdaMallet.load(self.model_odir + "/mallet.model")
        self.topicterms = self.model.get_topics() + 0.01
        self.doctopics = corpus2csc([i for i in self.model.load_document_topics()]).T

        # need corpus
        with open(self.model_odir + "/corpus.pickle", 'rb') as cp:
            self.corpus = pickle.load(cp)
        tdm = corpus2csc(self.corpus)
        self.doc_lens = np.asarray(tdm.sum(axis=0))
        self.word_counts = np.asarray(tdm.sum(axis=1))

        # need dictionary
        self.dictionary = Dictionary.load_from_text(self.model_odir + "/dict.txt")
        self.vocab = [t for t in self.dictionary.itervalues()]

        # need fnames
        with open(self.model_odir +  "/fnames.pickle", 'rb') as fp:
            self.fnames = pickle.load(fp)
