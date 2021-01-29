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
    subsets = None
    minwc = 2000000

    def __init__(self, meta, docs):
        self.meta = meta # pd dataframe
        self.docs = docs # pd Series
        self.meta["wordcounts"] = [len(d) for d in docs]
        self.meta["decade"] = [d[0:3] + "0" for d in meta["Date"]]
        self.embed_preprocessing()
        self.compute_embedding_subsets()

    def embed_preprocessing(self):
        """ given meta and docs, return pandas dataframe subset sentence """
        self.doc_sentences = [sentence_tokenize(d) for d in self.docs]
        self.doc_sentences = pd.Series(
                self.doc_sentences, index=self.docs.index)
        self.doc_sentences = self.doc_sentences.apply(
                lambda x: [normalize_text(s) for s in x])

    def compute_embedding_subsets(self):
        """ given meta return, series qid, subset 

        ["subsets dataframe: type, name, [ids]"

        """
        decades_inds = self.meta.groupby("decade").indices
        decades_inds = {d:[decades_inds[d], "decade"] for d in decades_inds.keys()}

        locations = self.meta.groupby("Location").sum("wordcounts")
        locations_inds = self.meta.groupby("Location").indices
        locations = list(locations[locations["wordcounts"] >= self.minwc].index)
        locations_inds = {l:[locations_inds[l], "location"] for l in locations}

        # decades, authors, locations
        authors_inds = list_group_by(self.meta["Author"])
        wcs = []
        for k,v in authors_inds.items():
            wcs.append(self.meta["wordcounts"][v].sum())
        authors = pd.Series(wcs, index=authors_inds.keys())
        authors = list(authors[authors >= self.minwc].index)
        authors_inds = {a:[authors_inds[a], "author"] for a in authors}

        combined = {**authors_inds, **locations_inds, **decades_inds} 
        combined = [[k,v[0], v[1]] for k,v in combined.items()]
        self.subsets = pd.DataFrame(combined, columns= ["name", "inds", "type"])
