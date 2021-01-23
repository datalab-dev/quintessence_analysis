"""
An object to hold the metadata and text documents (in several forms)
as they are pulled from the database, preprocessed and sent to the models 
Plan is to create one for the topic model data and one for the word2vec data.
"""
import pandas as pd

from quintessence.nlp import normalize_text
from quintessence.nlp import sentence_tokenize
from quintessence.nlp import list_group_by

class TMCorpus:
    meta = None
    docs = None

    def __init__(self, meta,docs):
        self.meta = meta
        self.docs = docs
        self.topic_model_preprocessing()

    def topic_model_preprocessing(self):
        normalized = [normalize_text(d) for d in self.docs]
        self.docs = pd.Series(normalized, index=self.docs.index)


class EmbedCorpus:
    meta = None
    docs = None
    sentences = None
    doc_sentences = None

    self.authors_inds = None
    self.locations_inds = None
    self.decades_inds = None

    def __init__(self, meta, docs):
        self.meta = meta # pd dataframe
        self.docs = docs # pd Series
        self.meta["wordcounts"] = [len(d) for d in docs]
        self.meta["decade"] = [d[0:3] + "0" for d in meta["Date"]]
        self.embed_preprocessing()

    def embed_preprocessing(self):
        """ given meta and docs, return pandas dataframe subset sentence """
        self.doc_sentences = [sentence_tokenize(d) for d in self.docs]
        self.doc_sentences = pd.Series(
                self.doc_sentences, index=self.docs.index)
        self.doc_sentences = self.doc_sentences.apply(lambda x:
                normalize_text(s) for s in x])


    def compute_embedding_subsets(self, minwc = 2000000):
        """ given meta return, series qid, subset """
        meta = self.meta

        # decades, authors, locations
        locations = meta.groupby("Location").sum("wordcounts")
        locations_inds = meta.groupby("Location").indices
        authors_inds = list_group_by(meta["Author"])
        wcs = []
        for k,v in authors_inds.items():
            wcs.append(meta["wordcounts"][v].sum())
        authors = pd.Series(wcs, index=authors_inds.keys())

        # keep only those with wordocunt > 2 million
        locations = list(locations[locations["wordcounts"] >= minwc].index)
        authors = list(authors[authors >= minwc].index)

        self.authors_inds = {a:authors_inds[a] for a in authors}
        self.locations_inds = {l:locations_inds[l] for l in locations}
        decades_inds = meta.groupby("decade").indices

    def get_sentences(self, subset):
        """ subset name and indices return sentence list """


