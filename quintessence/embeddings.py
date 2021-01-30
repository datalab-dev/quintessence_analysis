import os
import shutil

import gensim.models.word2vec

class Embeddings:
    def __init__(self, models_odir, sg=1, window=15, size=250, workers=4):
        self.models_odir = os.path.abspath(os.path.expanduser(models_odir))
        self.sg = sg
        self.window = window
        self.size = size
        self.workers = workers

        self.model = None
        self.subsets = {} 

    def create_model_dirs(self):
        """ if output directory exists, delete it and create new one """
        if os.path.isdir(self.models_odir):
            shutil.rmtree(self.models_odir, ignore_errors=True)

        os.mkdir(self.models_odir)
        dirs = ["author", "decade", "location"]
        dirs = [self.models_odir + "/" + d for d in dirs]
        for d in dirs:
            os.mkdir(d)

    def train_all(self, doc_sentences, subsets):
        """ train word2vec models """

        self.create_model_dirs()
        # foreach row
        for row in subsets.iterrows():
            row = row[1]
            flat = [s.split() for sentences in doc_sentences[row["inds"]] 
                    for s in sentences]
            model = gensim.models.Word2Vec(flat, sg=self.sg,
                    window = self.window, size = self.size,
                    workers = self.workers) 
            model.save(self.models_odir + "/" + row["type"] + 
                "/" + str(row["name"]) + ".model")
            self.subsets[str(row["name"])] = model

        sentences = [s.split() for sents in doc_sentences for s in sents]
        self.model = gensim.models.Word2Vec(sentences, sg=self.sg,
            window = self.window, size = self.size, workers = self.workers)
        model.save(self.models_odir + "/" + "full.model")


    def load_models(self):
        """
        Load models from directory
        """
        pass
