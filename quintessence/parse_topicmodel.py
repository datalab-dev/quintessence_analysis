import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.manifold import MDS

from quintessence.nlp import list_group_by

def create_topic_topterms(topicterms):
    """ 
    create topic.topterms
    _id: 0
    terms: [ ]
    scores: [ ]
    """
    tt = np.array(topicterms)
    inds = np.fliplr(tt.argsort(axis=1))

    docs = []
    for i,row in enumerate(inds):
        record = {
                "_id": i,
                "terms": list(topicterms.columns[row][0:100]),
                "scores": list(tt[i][row][0:100])
                }
        docs.append(record)
    return docs

def create_doc_topics (doctopics):
    """
    Create docs.topics data for mongo table

    doc.topics
    _id: 0,
    topics: [0.04 ...]

    returns list of dicts
    """
    docs = []
    for i,topics in enumerate(doctopics.to_records()):
        docs.append({'_id': i, 'topics': list(topics)})
    return docs

def create_topic_terms (topicterms):
    """
    Create topic.terms data for mongo table

    topic.terms
    _id: 0,
    terms: ["abate", ... ],
    scores: [0.01, ... ],
    ]

    returns list of dicts
    """

    terms = list(topicterms.columns)
    docs = []
    for i,scores in enumerate(topicterms.to_records(index=False)):
        docs.append({'_id': i, 'terms': terms, 'scores': list(scores)})
    return docs

def create_topics (corpusdf, doctopics, dtm, topicterms):
    """
    Create topics data for mongo table

    Create topics
    _id: 0,
    proportion: 0.0294,
    x: -0.13,
    y: 0.115,
    authors: [...],
    locations: [...],
    keywords: [...],
    dates: [...],
    topDocs: [1, 5, 345, 657, 34503]

    returns list of dicts
    """

    doc_lens = dtm.sum(axis=1) # row sums

    proportions = compute_proportions(doctopics, doc_lens)
    coordinates = compute_coordinates(topicterms)
    topdocs = compute_top_docs(doctopics)

    subsets = subset_proportions(corpusdf, doctopics, doc_lens)
    authors = subsets[0]
    locations = subsets[1]
    keywords = subsets[2]
    dates = subsets[3]

    # foreach topic
    docs = []
    for i in range(doctopics.shape[1]):
        docs.append({'_id': i,
            'proportion': float(proportions[i]),
            'x': float(coordinates[i][0]),
            'y': float(coordinates[i][1]),
            'topAuthors':  list(
                authors[i].sort_values(ascending=False)[0:10].index),
            'topLocations': list(
                locations[i].sort_values(ascending=False)[0:10].index),
            'topKeywords': list(
                keywords[i].sort_values(ascending=False)[0:10].index),
            'years': dates[i].to_dict(),
            'topDocs': [int(d) for d in topdocs[0:10, i]]})
    return docs


def compute_proportions(doctopics, doc_lens):
    """ Compute corpus wide topic proportions """
    weighted = doctopics.multiply(doc_lens, axis=0) # multiply dt by doc_lens
    colsums = weighted.sum(axis=0)
    return colsums / weighted.values.sum()

def compute_coordinates(topicterms):
    """ 
    Compute x and y coordinates for topics using multidimensional scaling of 
    topic terms matrix 
    """

    tt = np.array(topicterms)
    dists = np.zeros(shape=(tt.shape[0], tt.shape[0]))

    for i in range(tt.shape[0]):
        for j in range(i + 1, tt.shape[0]):
            dists[i][j] = jensenshannon(tt[i], tt[j])
    dists = dists + dists.T

    return MDS(n_components=2, 
            dissimilarity = "precomputed").fit_transform(dists)

def compute_top_docs(doctopics):
    """ returns ndarray rows are topic values are doc ids """
    dt = np.array(doctopics)
    topdocs = dt.argsort(axis=0)[::-1]
    return topdocs

    return topterms


def compute_topic_proportion (group_indices, doctopics, doc_lens):
    """
    For the given group indices, compute proportion for each 
    unique entry in the subset (e.g if subset = author, then each entry
    is a unique author

    returns pandas dataframe, rows are unique values, columns are topics,
    values are mean nonzero proportion
    """
    names = list(group_indices.keys())
    res = np.zeros((len(names), doctopics.shape[1]))

    i = 0
    for n,indices in group_indices.items():
        dt = doctopics.loc[indices]
        dl = doc_lens[indices]
        res[i] = compute_proportions(dt, dl)
        i += 1

    return pd.DataFrame(res, index=names)

def subset_proportions(corpus, doctopics, doc_lens):
    """  
    for each metadata grouping (e.g London, 'John Donne etc) 
    compute topic prorportions

    return list of dataframes
    """
    # get inds for each unique value
    authors_inds = list_group_by(corpus["Author"])
    locations_inds = corpus.groupby("Location").groups
    keywords_inds = list_group_by(corpus["Keywords"])
    dates_inds = corpus.groupby("Date").groups

    # compute mean nonzero proportion foreach subset
    authors = compute_topic_proportion(authors_inds, doctopics, doc_lens)
    locations = compute_topic_proportion(locations_inds, doctopics, doc_lens)
    keywords = compute_topic_proportion(keywords_inds, doctopics, doc_lens)
    dates = compute_topic_proportion(dates_inds, doctopics, doc_lens)
    return [authors, locations, keywords, dates]
