import copy

import pandas as pd
from gensim.models.word2vec import Word2Vec

from quintessence.nlp import procrustes_alignment

def get_vocab(full):
    """ return dictionary id:term """
    vocab = sorted(list(full.wv.vocab))
    return {term:i for i,term in enumerate(vocab)}

def create_terms(full, vocab):
    """ create terms collection docs

    terms
    termId: 0
    term: "abbot"
    neighbors:
    scores:
    """

    docs = []
    for term,termId in vocab.items():
        terms, scores = zip(*full.most_similar(term))
        record = {
                "termId" : termId,
                "term" : term,
                "freq" : full.wv.vocab[term].count,
                "neighbors": [vocab[t] for t in terms],
                "scores": list(scores)
                } 
        docs.append(record)
    return docs

def create_subsets(subsets):
    """ create subsets collection docs

    subsets
    subsetId:
    type:
    subset:

    """
    index = 0
    docs = []
    for name,_,stype in subsets:
        record = {
                "subsetId": index,
                "type": stype,
                "subset": name.split('.')[0]
                }
        docs.append(record)
        index+=1
    return docs


def create_nearest_neighbors(subset, vocab, n=10)
    """ Create collection of nearest neighbors for a subset

    terms.subset
    termId:
    freq:
    neighbors: [ ] # intergers
    scores: [ ] # floats
    """

    name = subset[0]
    model = subset[1]
    stype = subset[2]

    for term,termId in vocab.items():
        # if term in model:
        if term in model.vocab:
            freq = model.wv.vocab[term].count
            terms, scores = zip(*model.most_similar(term))
            terms = [vocab[t] for t in terms]
            scores = list(scores)
        else:
            freq = 0
            terms = []
            scores = []

        record = {
                "termId" : termId,
                "freq": freq,
                "neighbors": [vocab[t] for t in terms],
                "scores": list(scores)
                } 

        docs.append(record)
    return docs
            

def create_similarity_over_time(decades, vocab):
    """
    Create terms.timeseries collection docs

    terms.timeseries
    termId: 0,
    similarities: [0.90, 0.84, 0.89, ...] floats
    """

    docs = []
    # sort decades from smallest to largest (earliest to last)
    # base model is the largest
    decades.sort(key=lambda tup: tup[0])

    ## compute alignments
    alignments = []
    for d in decades:
        # need to do a deepcopy here on the base model because otherwise
        # it will be altered by each alignment since python is pass
        # by object reference (in this case behaving like pass by reference)
        # dont need to deepcopy the others, since its fine if they
        # are altered in the embeddings class, but not sure what will be changed
        # in the future, so...
        base,other = procrustes_alignment(copy.deepcopy(decades[-1][1]),
                copy.deepcopy(d[1]))
        alignments.append((base, other))

    # for each term

    #    init similaries list
    #    get termId

    #      for each subset (strating 1400 - 1700)
    #           compute similarity to same term in 1700; append to list

    #  create record, append to docs
    return docs

