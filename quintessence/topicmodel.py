from ast import literal_eval
import os
import pathlib
import shutil
import pickle

from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from joblib import Parallel
from joblib import delayed
import numpy as np
import pandas as pd

from quintessence.nlp import normalize_text

class TopicModel:
    def __init__(self, model_dir):
        """
        Initialize a TopicModel class.
        Either to use for training models
        or to load in a model trained by this class.
        """
        # dirname for mallet model, temp files, and dictionary ...
        # note if expanduser not set then it will incorrectly handle ~ 
        # when setting absolute; for example test os.path.abspath(~/Documents)
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))
        self.doctopics = pd.DataFrame()
        self.topicterms = pd.DataFrame()
        self.dtm = pd.DataFrame()
        self.meta = pd.DataFrame()


    def preprocess(self, texts, workers):
        """ called in train """
        print("normalize")
        normalized = Parallel(n_jobs = workers)(delayed(
            normalize_text)(d) for d in texts)
        print("construct dictionary")
        dictionary = Dictionary(normalized)
        dictionary.filter_extremes(no_below=int(0.01 * len(normalized)), no_above=0.8)
        print("make dtm")
        #dtm = Parallel(n_jobs = workers)(delayed(
        #    dictionary.doc2bow)(doc) for doc in normalized)
        dtm = [dictionary.doc2bow(doc) for doc in normalized] # for some reason this is faster as list comprehension than 'Parallel'
        return dtm, dictionary

    def train(self, corpusdf, mallet_path, num_topics, workers=4, mallet_workers=24):
        """
        Train topic model using mallet.

        Docs is pandas series, index is ids, and values are normalized strings
        mallet_path is a string containing path to mallet binary
            ex: '~/mallet-2.0.8/bin/mallet'
        num_topics is int

        Saves all outputs as properties of the class instance, as well as to files in self.model_dir path
        """
        if os.path.isdir(self.model_dir):
            shutil.rmtree(self.model_dir, ignore_errors=True)
        os.makedirs(self.model_dir)

        mallet_path = os.path.abspath(os.path.expanduser(mallet_path)) 
        self.meta = corpusdf.loc[:, corpusdf.columns != 'docs'] 

        print("preprocessing")
        self.dtm,self.dictionary = self.preprocess(corpusdf["docs"], workers)

        # train model
        # mallet is a dummy? add / to prefix...
        print("training")
        self.model = LdaMallet(mallet_path,
                corpus=self.dtm, prefix = self.model_dir + "/",
                          num_topics=num_topics, id2word=self.dictionary,
                          workers = mallet_workers)

        vocab = [t for t in self.dictionary.itervalues()]
        fnames = corpusdf.index

        print("extracting topicterms, doctopics, dtm")
        self.topicterms = self.model.get_topics()
        self.topicterms = pd.DataFrame(data=self.topicterms,
                index=range(num_topics), columns=vocab)
        self.doctopics = corpus2csc([i for i in self.model.load_document_topics()]).T
        self.doctopics = self.doctopics.toarray()
        self.doctopics = pd.DataFrame(data=self.doctopics, 
                index=fnames, columns = range(num_topics))
        self.dtm = corpus2csc(self.dtm).T
        self.dtm = self.dtm.toarray()
        self.dtm = pd.DataFrame(data=self.dtm, index=corpusdf.index, columns = vocab)


        # save model, dtm, topicterms, doctopics
        print("saving to disk")
        self.model.save(self.model_dir + "/mallet.model") 
        self.meta.to_csv(self.model_dir + "/meta.csv")
        self.topicterms.to_csv(self.model_dir + "/tt.csv")
        self.doctopics.to_csv(self.model_dir + "/dt.csv")
        self.dtm.to_csv(self.model_dir + "/dtm.csv")


    def load_model(self):
        """
        Load a previously saved model and the other files that should be in the odir
        updates all the class instance properties (vocab, doctopics ...)
        Does not require a mallet path
        """
        self.model = LdaMallet.load(self.model_dir + "/mallet.model")
        self.meta = pd.read_csv(self.model_dir + "/meta.csv", index_col="_id", converters={
            "Keywords": literal_eval, 
            "Author": literal_eval, 
            "Languages": literal_eval})
        self.topicterms = pd.read_csv(self.model_dir + "/tt.csv", index_col=0)
        self.doctopics = pd.read_csv(self.model_dir + "/dt.csv", index_col="_id")
        self.dtm = pd.read_csv(self.model_dir + "/dtm.csv", index_col="_id")
