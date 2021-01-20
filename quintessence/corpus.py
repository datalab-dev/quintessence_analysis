"""
An object to hold the metadata and text documents (in several forms)
as they are pulled from the database, preprocessed and sent to the models 
Plan is to create one for the topic model data and one for the word2vec data.
"""
import pandas as pd

from quintessence.nlp import normalize_text
from quintessence.nlp import sentence_tokenize

class Corpus:
    sentences = []
    doc_sentences = []

    def __init__(self, meta, docs):
        self.meta = meta # pd dataframe
        self.docs = docs # pd Series

    def topic_model_preprocessing(self):
        normalized = [normalize_text(d) for d in self.docs]
        self.docs = pd.Series(normalized, index=self.docs.index)

    def embed_preprocessing(self):
        self.doc_sentences = [sentence_tokenize(d) for d in self.docs]
        self.sentences = [normalize_text(s).split() 
                for sents in self.doc_sentences
                for s in sents]
