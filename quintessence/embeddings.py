import os
import shutil
import pathlib

import gensim.models.word2vec

class Embeddings:
    def __init__(self, models_dir):
        self.models_dir = os.path.abspath(os.path.expanduser(models_dir))

        self.model = None
        self.subsets = []

    def create_model_dirs(self):
        """ if output directory exists, delete it and create new one """
        if os.path.isdir(self.models_dir):
            shutil.rmtree(self.models_dir, ignore_errors=True)

        os.mkdir(self.models_dir)
        dirs = ["author", "decade", "location"]
        dirs = [self.models_dir + "/" + d for d in dirs]
        for d in dirs:
            os.mkdir(d)

    def train_all(self, 
            doc_sentences,
            subsets,
            sg = 1,
            window = 15,
            size = 250,
            workers = 4):
        """ train word2vec models """

        def make_filename(row):
            name = str(row["name"]).replace(" ", "_")
            fname = self.models_dir + "/" + row["type"] + "/" + name + ".model"
            return fname

        self.create_model_dirs()
        # foreach row
        for _,row in subsets.iterrows():
            flat = [s.split() for sentences in doc_sentences[row["inds"]] 
                    for s in sentences]
            model = gensim.models.Word2Vec(flat, sg=sg,
                    window = window, size = size,
                    workers = workers) 
            model.save(make_filename(row))
            self.subsets.append(([str(row["name"]).replace(" ","_")], 
                    model, row["type"]))

        sentences = [s.split() for sents in doc_sentences for s in sents]
        self.model = gensim.models.Word2Vec(sentences, sg=sg,
            window = window, size = size, workers = workers)
        model.save(self.models_dir + "/" + "full.model")


    def load_models(self):
        """
        Load models from directory
        ./data/embeddings
            full.model
            location/*.model
            author/*.model
            decade/*.model

        """

        self.model = None
        self.subsets = []

        # full
        self.model = gensim.models.Word2Vec.load(self.models_dir + "/full.model")

        #subsets
        models = list(pathlib.Path("./data/embeddings").rglob("*.model"))
        for m in models:
            name = m.name
            mtype = m.parent.name
            if name != "full.model":
                self.subsets.append((name, 
                        gensim.models.Word2Vec.load(str(m)), mtype))
