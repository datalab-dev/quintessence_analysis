from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary

class TopicModel:
    def __init__(self, model_path, mallet_path=None, num_topics=None):
        self.model_path = model_path # fname for mallet model
        self.mallet_path = mallet_path # ex: '~/mallet-2.0.8/bin/mallet'
        self.num_topics = num_topics
        self.model = None # LdaMallet obj

    def train(self, docs):
        """
        Train topic model using mallet.
        """
        docs = [doc.split() for doc in docs]
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=int(0.01 * len(docs)), no_above=0.8) # prune
        corpus = [dictionary.doc2bow(doc) for doc in docs]

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
