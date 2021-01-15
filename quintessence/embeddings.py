import string
import time

import gensim.models.word2vec
from joblib import Parallel, delayed
import nltk
import pymongo

# TODO:
# split into two classes Embeddings and Embeddings Controller
# Embeddings Controller is logic for which embeddings to run (e.g compute
# subsets and so on whereas embeddings just trains a model and saves 
# the output

class Embeddings:
    def __init__(self, model_fpath, sg=1, window=15, size=250, workers=4):
        self.model_fpath = model_fpath
        self.sg = sg
        self.window = window
        self.size = size
        self.workers = workers

    def train(self, sentences):
        """
        Train word2vec model for set of sentences.

        Args:
            sentences: list of lists of words
            sg: training algorithm (1 for skipgram otherwise CBOW)
            window: max distance between current and predicted word in sentence
            size: dimensionality of word vectors
            workers: number of workers for training parallelization
        """
        # gensim word2vec
        start = time.time()
        model = gensim.models.Word2Vec(sentences, 
                sg=self.sg, window=self.window,
                size=self.size, workers=self.workers)
        self.model = model

        print(f"trained model in {(time.time() - start) / 60} minutes")
        model.save(self.model_fpath)

    def load_model(self):
        """
        Load model from npy file.
        """
        self.model = gensim.models.Word2Vec.load(self.model_fpath)

