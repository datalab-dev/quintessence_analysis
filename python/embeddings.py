import time
import pymongo
from joblib import Parallel, delayed
import nltk
import string

from mongo import db


class Embeddings:
    def __init__(self, model_dir, stopwords_path=None):
        self.model_dir = Path(model_dir)
        self.stopwords = open(stopwords_path).read().splitlines()
        # self.authors = open(authors_path).read().splitlines()
        # self.locations = open(locations_path).read().splitlines()
        self.sentences = self.load_corpus()

    def get_sentences(self, doc_content):
        """
        Given standardized text return list of cleaned and tokenized sentences.
        """
        cleaned = []
        doc_content = doc_content.replace("\t", " ")
        sentences = nltk.sent_tokenize(doc_content)

        for s in sentences:
            s = s.lower()
            s = s.replace('|', ' ')
            s = s.translate(str.maketrans('', '', string.punctuation))
            words = [w for w in s.split() if w not in self.stopwords]
            cleaned.append(words)

        return cleaned

    def load_corpus(self):
        """
        Clean corpus and save as sentences.
        """
        # load the corpus from mongo
        cursor = db.docs.find(projection={'standardized': 1}).sort('_id', 1)
        docs = cursor[:]

        # split and clean
        sentences = Parallel(n_jobs=80)(
            delayed(get_sentences)(doc['standardized']) for doc in docs)

        return sentences

    def train(self, sentences, sg=1, window=15, size=250, workers=80):
        """
        Train word2vec model for set of sentences.

        Args:
            sg: training algorithm (1 for skipgram otherwise CBOW)
            window: max distance between current and predicted word in sentence
            size: dimensionality of word vectors
            workers: number of workers for training parallelization
        """
        # gensim word2vec
        start = time.time()
        model = gensim.models.Word2Vec(sentences, sg=sg, window=window,
                                       size=size, workers=workers)
        print(f"trained model in {(time.time() - start) / 60} minutes")

        # TODO save model
        model.save()

    def train_full(self):
        """
        Train word2vec model for full corpus.
        """
        train(self.sentences)

    def train_decades(self):
        """
        Train word2vec model for each decade.
        """
        pipeline = [
            {'$group': {
                '_id': {'$floor': {'$divide': ['$date', 10]}},
                'qids': {'$push': '$_id'}
            }}
        ]
        cursor = db.docs.metadata.aggregate(pipeline=pipeline)
        for group in cursor:
            decade = int(obj['_id'])
            subset = [self.sentences[i] for i in obj['qids']]
            train(subset)

    def train_authors(self):
        """
        Train word2vec model for each author in self.authors.
        """
        pipeline = [
            {'$match': {'$_id': {'$in': self.authors}}}
            {'$unwind': '$author'},
            {'$group': {
                '_id': "$author",
                'qids': {'$push': '$_id'},
                'wordCount': {'$sum': '$wordCount'}
            }},
            {'$match': {'wordCount': {'$gte': 1990000}}}
        ]
        cursor = db.docs.metadata.aggregate(pipeline=pipeline)
        for group in cursor:
            decade = int(obj['_id'])
            subset = [self.sentences[i] for i in obj['qids']]
            train(subset)

    def train_locations(self):
        """
        Train word2vec model for each locatio in self.locations.
        """
        pipeline = [
            {'$match': {'$_id': {'$in': self.locations}}}
            {'$unwind': '$location'},
            {'$group': {
                '_id': "$location",
                'qids': {'$push': '$_id'},
                'wordCount': {'$sum': '$wordCount'}
            }},
            {'$match': {'wordCount': {'$gte': 2000000}}}
        ]
        cursor = db.docs.metadata.aggregate(pipeline=pipeline)
        for group in cursor:
            decade = int(obj['_id'])
            subset = [self.sentences[i] for i in obj['qids']]
            train(subset)

    def load_model(self):
        """Load model from npy file."""
        pass

    def align(self):
        """Create terms.timeseries.""""
        pass

    def nn(self):
        """Create terms.neighbors.""""
        pass

    def freq(self):
        """"Create topics.frequencies."""
        pass

    def update(self):
        self.train_full()
        self.train_decades()
        self.train_authors()
        self.train_locations()
        self.align()
        self.nn()
        self.freq()
