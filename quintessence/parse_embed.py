import copy

from gensim.models.word2vec import Word2Vec
import pandas as pd
from scipy import spatial

from quintessence.alignment import procrustes_alignment

def get_all_vocab(full, subsets, decades):
    v = set(full.wv.vocab)
    for s in subsets:
        v = v.union(v, set(s[1].wv.vocab))
    for d in decades:
        v = v.union(v, set(d[1].wv.vocab))
    vocab = sorted(list(v))
    return {term:i for i,term in enumerate(vocab)}

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
        terms = []
        scores = []
        # since terms now include all the terms accross all models, it might not appear in the full model
        # since the full model only includes the top 100k terms, and full vocab has 400k terms...
        if term in full.wv.vocab:
            terms, scores = zip(*full.most_similar(term))
        record = {
                "termId" : termId,
                "term" : term,
                "neighbors": list(terms),
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
                "subset": name
                }
        docs.append(record)
        index+=1
    return docs


def create_nearest_neighbors(subset, vocab, n=10):
    """ Create collection of nearest neighbors for a subset
    Subset can be decade < ('date', 'model) > or 
    generic subset < ('name', model, 'type') >

    terms.subset
    termId:
    freq:
    neighbors: [ ] # intergers
    scores: [ ] # floats
    """

    name = subset[0]
    model = subset[1]

    docs = []
    for term in model.wv.vocab.keys():
        termId = vocab[term]
        freq = model.wv.vocab[term].count
        terms, scores = zip(*model.wv.most_similar(term))
        record = {
                "termId" : termId,
                "freq": freq,
                "neighbors": list(terms),
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

    # remove pre 1470 and post 1700
    decades = [d for d in decades if int(d[0]) < 1710 and int(d[0]) > 1460]

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

    # for each term in 1700 model...
    for term in decades[-1][1].wv.vocab.keys():
        similarities = [None for i in range(len(alignments))]

        for i,align in enumerate(alignments):
            if term in align[0].wv.vocab.keys():
                ind = align[0].wv.vocab[term].index
                similarities[i] = 1 - spatial.distance.cosine(
                        align[0].wv.vectors[ind,],
                        align[1].wv.vectors[ind,])

        #  create record, append to docs
        record = {
                "termId": vocab[term],
                "similarities": similarities
                }
        docs.append(record)
    return docs

