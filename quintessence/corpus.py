"""
An object to hold the metadata and text documents (in several forms)
as they are pulled from the database, preprocessed and sent to the models 
Plan is to create one for the topic model data and one for the word2vec data.
"""
from quintessence.nlp import normalize_text
from quintessence.nlp import sentence_tokenize

class Corpus:
    def __init__(self, meta, doclist, ids=range(0,len(doclist))):
        self.meta = meta
        self.docs = doclist
        self.ids = ids
        self.sentences = []
        self.doc_sentences = []

    def topic_model_preprocessing():
        self.docs = [normalize_text(d) for d in self.docs]

    def embed_preprocessing():
        self.doc_sentences = [sentence_tokenzie(d) for d in docs]
        self.sentences = [normalize_text(s).split() for sents in doc_sentences
                for s in sents]
