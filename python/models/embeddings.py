import time
import pymongo
from joblib import Parallel, delayed
import nltk
import string

from utils.mongo import db
import utils.nlp


class Embeddings:
    def __init__(self, model_dir, stopwords_path=None):
        self.model_dir = model_dir
        self.stopwords = open(stopwords_path).read().splitlines()
        self.sentences = self.load_corpus()

    def load_corpus(self):
        """
        Clean corpus and save as sentences.
        """
        # load the corpus from mongo
        cursor = db.docs.find(projection={'standardized': 1}).sort('_id', 1)
        docs = cursor[:]

        # split and clean
        sentences = Parallel(n_jobs=80)(
            delayed(nlp.tokenize)(doc['standardized']) for doc in docs)

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
        """
        Load model from npy file.
        """
        pass

    def align(self):
        """
        Create terms.timeseries.
        """
        # TODO add smart_procrustes_align_gensim (maybe in utils?)

        # TODO load models  + model names from somwhere
        # TODO t0 = model for 1700

        t0.init_sims()

        dists = {}
        for model in models:
            model.init_sims()
            aligned = smart_procrustes_align_gensim(t0, model)
            for term in aligned.wv.vocab.keys():
                base = t0.wv[term]
                vec = aligned.wv[term]
                similarity = 1 - scipy.spatial.distance.cosine(base, vec)
                if v not in dists:
                    dists[term] = np.zeros(len(models), dtype=float)
                dists[term][i] = similarity

        docs = [{'_id': term, 'timeseries': dists[term]} for term in dists]
        db['terms.timeseries'].remove()
        db['terms.timeseries'].insert_many(docs)

    def nn(self, model_type, n=20):
        """
        Create terms.neighbors.

        Args:
            model_type: {'full', 'decades', 'authors', 'locations'}
            n: the number of top nearest neighbors to store
        """
        # TODO load models  + model names from somwhere

        if model_type == 'full':
            termdict = {'full': {'neighbors': [], 'scores': []}}
        else:
            termdict = {}
            termdict[model_type] = {
                name: {'neighbors': [], 'scores': []} for name in model_names
            }

        for model in models:
            for term in model.wv.vocab.keys:
                if term not in nndict:
                    nndict[term] = termdict
                    nndict[term]['_id'] = term
                results = model.wv.vocab.keys.most_similar(word, topn=n)
                for result in results:
                    nn = nndict[term][model_type]
                    if model_type != 'full'
                        nn = nn[model_name]
                    nn.neighbors.append(result[0])
                    nn.scores.append(result[1])

        docs = list(nndict.values())
        update = {'$set': docs}
        db['terms.neighbors'].update_many({}, update)

    def freq(self, model_type):
        """
        Create terms.frequencies.

        Args:
            model_type: {'full', 'decades', 'authors', 'locations'}
        """
        # TODO load models  + model names from somwhere

        if model_type == 'full':
            termdict = {'full': {'freq': None, 'relFreq': None}}
        else:
            termdict = {}
            termdict[model_type] = {
                name: {'freq': None, 'relFreq': None} for name in model_names
            }

        for model in models:
            total = sum(model.wv.vocab[term].count for term, vocab_obj in
                        model_wv.vocab.items())
            for term, vocab_obj in model_wv.vocab.items():
                if term not in nndict:
                    freqdict[term] = termdict
                    freqdict[term]['_id'] = term
                count = model.wv.vocab[term].count
                f = nndict[term][model_type]
                if model_type != 'full'
                    f = nn[model_name]
                f.freq = count
                f.relFreq = count / total

        docs = list(freqdict.values())
        update = {'$set': docs}
        db['terms.frequencies'].update_many({}, update)

    def update(self):
        self.train_type(type='full')
        self.train_type(type='decades')
        self.train_type(type='authors')
        self.train_type(type='locations')
        self.align()
        self.nn(type='full')
        self.nn(type='decades')
        self.nn(type='authors')
        self.nn(type='locations')
        self.freq()
