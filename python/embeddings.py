import subprocess


class Embeddings:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)

    def load_corpus(self):
        """Clean corpus and save as sentences."""
        # TODO load the corpus from mongo

        # TODO split and clean

        pass


    def train(self, sentences):
        """Train word2vec model for set of sentences."""
        # TODO gensim word2vec

        # TODO save model

        pass


    def train_full(self):
        pass


    def train_decades(self):
        pass


    def train_authors(self):
        pass


    def train_locations(self):
        pass


    def load_model(self):
        """Load model from npy files."""
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
