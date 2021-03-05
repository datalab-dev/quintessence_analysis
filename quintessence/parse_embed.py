from collections import namedtuple
import copy

from gensim.models.word2vec import Word2Vec
import pandas as pd
from scipy import spatial

from quintessence.alignment import procrustes_alignment

def create_embeddings_datamodel(full, subsets, decades):
    collections = {}

    vocab = get_all_vocab(full, subsets, decades)

    # terms.subsets
    print("terms.subsets")
    collections["terms.subsets"] = create_subsets(subsets)
    
    # terms.neighbors
    print("terms.neighbors")
    collections["terms.neighbors"] = create_nearest_neighbors(
            full, subsets, decades, vocab)
    
    # terms.timeseries
    #    for each term in 1700 model, time series data
    print("terms.timeseries")
    collections["terms.timeseries"] = create_similarity_over_time(decades)

    return collections

def create_subsets(subsets):
    """ create terms.subsets collection docs

    terms.subsets
    type:
    subset:

    """
    index = 0
    docs = []
    for name,_,stype in subsets:
        record = {
                "_id": name
                "type": stype,
                }
        docs.append(record)
        index+=1
    return docs


def create_nearest_neighbors(full, subsets, decades, vocab):

    def create_neighbors_record(term, model):
        terms, scores = zip(*model.wv.most_similar(term))
        return {
                "terms": list(terms),
                "scores": list(scores),
                "freqs": [model.wv.vocab[t].count for t in terms]
                "freq": model.wv.vocab[term].count
                }

    docs = []
    for term in vocab:
        record = {}
        record["_id"] = term

        # for full
        if term in full.wv.vocab.keys():
            record["full"] = create_neighbors_record(term, full)

        # for subsets
        for s in subsets:
            if term in s[1].wv.vocab.keys():

                if s[2] == "location":
                    if "locations" not in record.keys():
                        record["locations"] = {}
                    record["locations"][s[0]] = create_neighbors_record(term, s[1])

                if s[2] == "author":
                    if "authors" not in record.keys():
                        record["authors"] = {} 
                    record["authors"][s[0]]["freq"] = freq

        # for decades
        for d in decades:
            if term in d[1].wv.vocab.keys():
                if "decades" not in record.keys():
                    record["decades"] = {}
                record["decades"][d[0]] = create_neighbors_record(term, d[1])


        docs.append(record)
    return docs
            

def create_similarity_over_time(decades):
    """
    Create terms.timeseries collection docs
    only contains terms in 1700 model

    terms.timeseries
    _id: "history",
    similarities: [0.90, 0.84, 0.89, ...] floats
    decades: [1700, 1690, 1680 ...] strings
    """

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
        alignments.append((base, other, d[0]))

    # for each term in 1700 model...
    docs = []
    for term in decades[-1][1].wv.vocab.keys():
        similarities = [None for i in range(len(alignments))]
        dec = []

        for i,align in enumerate(alignments):
            if term in align[0].wv.vocab.keys():
                ind = align[0].wv.vocab[term].index
                similarities[i] = 1 - spatial.distance.cosine(
                        align[0].wv.vectors[ind,],
                        align[1].wv.vectors[ind,])
                dec.append(align[2])

        #  create record, append to docs
        record = {
                "_id": term,
                "timeseries": similarities,
                "decades": dec
                }
        docs.append(record)
    return docs


def get_all_vocab(full, subsets, decades):
    v = set(full.wv.vocab)
    for s in subsets:
        v = v.union(v, set(s[1].wv.vocab))
    for d in decades:
        v = v.union(v, set(d[1].wv.vocab))
    vocab = sorted(list(v))
    return vocab
