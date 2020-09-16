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

    def train_type(self, model_type):
        """
        Train word2vec model based on specified groupign.

        Args:
            model_type: {'full', 'decades', 'authors', 'locations'}
        """
        if model_type == 'full':
            train(self.sentences)
            return

        if model_type == 'decades':
            pipeline = [
                {'$group': {
                    '_id': {'$floor': {'$divide': ['$date', 10]}},
                    'qids': {'$push': '$_id'}
                }}
            ]
        elif model_type == 'authors':
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
        elif model_type == 'locations':
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
        else:
            raise ValueError("invalid model type")

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

    def nn(self, model_type, n=20):
        """
        Create terms.neighbors.

        Args:
            model_type: {'full', 'decades', 'authors', 'locations'}
            n: the number of top nearest neighbors to store
        """"
        # TODO load models  + model names from somwhere

        vocab = model.wv.vocab.keys
        termdict = {}
        termdict[model_type] = {
            model_name: {neighbors=[], scores=[]} for model_name in model_names
        }
        nndict = {term: termdict for term in vocab}

        for model in models:
            for term in vocab:
                results = vocab.most_similar(word, topn=n)
                for result in results:
                    nn = nndict[term][model_type][model_name]
                    nn.neighbors.append(result[0])
                    nn.scores.append(result[1])

        update = {'$set': nndict}
        db['terms.neighbors'].update_many({}, update)

    def freq(self):
        """"Create topics.frequencies."""
        pass

    def update(self):
        self.train_type(type='full')
        self.train_type(type='decades')
        self.train_type(type='authors')
        self.train_type(type='locations')
        self.align()
        self.nn(type='decades')
        self.nn(type='authors')
        self.nn(type='locations')
        self.freq()
