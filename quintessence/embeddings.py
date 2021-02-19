import os
import pathlib
import shutil

import gensim.models.word2vec
from joblib import delayed
from joblib import Parallel
import numpy as np
import pandas as pd

from quintessence.nlp import list_group_by
from quintessence.nlp import normalize_text
from quintessence.nlp import sentence_tokenize

class Embeddings:
    def __init__(self, models_dir):
        self.models_dir = os.path.abspath(os.path.expanduser(models_dir))

        self.full = None
        self.subsets = [] # list of tuples
        self.decades = [] #  list of tuples

    def create_model_dirs(self):
        """ if output directory exists, delete it and create new one """
        if os.path.isdir(self.models_dir):
            shutil.rmtree(self.models_dir, ignore_errors=True)

        os.makedirs(self.models_dir)
        dirs = ["author", "decade", "location"]
        dirs = [self.models_dir + "/" + d for d in dirs]
        for d in dirs:
            os.mkdir(d)


    def preprocessing(self, corpusdf, workers=4):
        """ corpusdf["docs"] -> normalized list (sentences)  of lists (words)"""

        def normalize_sentences(d):
            return [normalize_text(s) for s in d]

        docs = corpusdf["docs"]

        doc_sentences = Parallel(n_jobs=workers)(delayed(sentence_tokenize)(d)
                for d in docs)

        doc_sentences = Parallel(n_jobs=workers)(delayed(
            normalize_sentences)(d) for d in doc_sentences)

        corpusdf["docs"] = doc_sentences
        return corpusdf


    def compute_subsets(self, corpusdf, minwc=2000000):
        """ given meta return, series qid, subset 

        ["subsets dataframe: type, name, [ids]"

        """
        corpusdf["wordcounts"] = corpusdf["docs"].apply(lambda x: len(x.split()))
        corpusdf["decade"] = corpusdf["Date"].apply(lambda x: x[0:3] + "0")

        decades_inds = corpusdf.groupby("decade").groups
        decades_inds = {d:[decades_inds[d], "decade"] for d in decades_inds.keys()}

        locations = corpusdf.groupby("Location").sum("wordcounts")
        locations_inds = corpusdf.groupby("Location").groups
        locations = list(locations[locations["wordcounts"] >= minwc].index)
        locations_inds = {l:[locations_inds[l], "location"] for l in locations}

        # decades, authors, locations
        authors_inds = list_group_by(corpusdf["Author"])
        wcs = []
        for k,v in authors_inds.items():
            wcs.append(corpusdf["wordcounts"][v].sum())
        authors = pd.Series(wcs, index=authors_inds.keys())
        authors = list(authors[authors >= minwc].index)
        authors_inds = {a:[authors_inds[a], "author"] for a in authors}

        combined = {**authors_inds, **locations_inds, **decades_inds} 
        combined = [[k,v[0], v[1]] for k,v in combined.items()]
        subset_inds = pd.DataFrame(combined, columns= ["name", "inds", "type"])
        return subset_inds


    def train_all(self, 
            corpusdf,
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

        print("compute subsets") 
        subset_inds = self.compute_subsets(corpusdf)

        print("preprocess: sentence tokenize, normalize")
        corpusdf = self.preprocessing(corpusdf)

        # foreach subset
        print("train subset models")
        for _,row in subset_inds.iterrows():

            print("... " + str(row["name"]))
            flat = [s for sentences in corpusdf["docs"].loc[row["inds"]] 
                    for s in sentences]
            model = gensim.models.Word2Vec(flat, sg=sg,
                    window = window, size = size,
                    workers = workers) 
            model.save(make_filename(row))

            if row["type"] == "decade":
                self.decades.append((str(row["name"]), model))
            else:
                self.subsets.append((str(row["name"]).replace(" ","_"), 
                    model, row["type"]))

        print("train full model")
        sentences = [s for sents in corpusdf["docs"] for s in sents]
        self.model = gensim.models.Word2Vec(sentences, sg=sg,
            window = window, size = size, workers = workers)
        self.model.save(self.models_dir + "/" + "full.model")

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
        models = list(pathlib.Path(self.models_dir).rglob("*.model"))
        for m in models:
            name = m.name
            name = name.split('.')[0] # remove .model from name
            mtype = m.parent.name
            if name != "full":

                if mtype == "decade":
                    self.decades.append((name, 
                        gensim.models.Word2Vec.load(str(m))))
                else:
                    self.subsets.append((name, 
                        gensim.models.Word2Vec.load(str(m)), mtype))
