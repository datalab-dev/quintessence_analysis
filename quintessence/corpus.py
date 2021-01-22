"""
An object to hold the metadata and text documents (in several forms)
as they are pulled from the database, preprocessed and sent to the models 
Plan is to create one for the topic model data and one for the word2vec data.
"""
import pandas as pd

from quintessence.nlp import normalize_text
from quintessence.nlp import sentence_tokenize
from quintessence.nlp import list_group_by

class Corpus:
    meta = None
    docs = None
    sentences = None
    doc_sentences = None

    def __init__(self, meta, docs):
        self.meta = meta # pd dataframe
        self.docs = docs # pd Series

    def topic_model_preprocessing(self):
        normalized = [normalize_text(d) for d in self.docs]
        self.docs = pd.Series(normalized, index=self.docs.index)

    def embed_preprocessing(self):
        """ given meta and docs, return pandas dataframe subset sentence """
        self.doc_sentences = [sentence_tokenize(d) for d in self.docs]
        self.doc_sentences = pd.Series(
                self.doc_sentences, index=self.docs.index)

        subsets = compute_embedding_subsets()

        self.sentences = [normalize_text(s).split() 
                for sents in self.doc_sentences
                for s in sents]

    def compute_embedding_subsets(self):
        """ given meta return, series qid, subset """
        meta = self.meta
        docs = self.docs
        meta["wordcounts"] = [len(d) for d in docs]
        meta["decade"] = [d[0:3] + "0" for d in meta["Date"]]

        # decades, authors, locations
        decades = meta.groupby("decade").sum("wordcounts")
        locations = meta.groupby("Location").sum("wordcounts")
        authors_inds = list_group_by(meta["Author"])
        wcs = []
        for k,v in authors_inds.items():
            wcs.append(meta["wordcounts"][v].sum())
        authors = pd.Series(wcs, index=authors_inds.keys())

        # keep only those with wordocunt > 2 million
        decades = list(decades[decades["wordcounts"] > 100000].index)
        locations = list(locations[locations["wordcounts"] > 100000].index)
        authors = list(authors[authors > 100000].index)

        pass

