import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.manifold import MDS

from quintessence.nlp import list_group_by

def create_doc_topics (doctopics):
    """
    Create docs.topics data for mongo table

    doc.topics
    _id: 0,
    topics: [
       {"topicId": 0, "probability": 0.05},
       {"topicId": 1, "probability": 0.08}, ...
      ]

    returns list of dicts
    """
    doctopics = doctopics.todense().A
    docs = []
    for qid, row in enumerate(doctopics):
        topics = []
        for topic, probability in enumerate(row):
            topics.append({'topicId': topic, 'probability': probability})
        docs.append({'_id': qid, 'topics': topics})
    return docs

def create_topic_terms (topicterms, dictionary):
    """
    Create topic.terms data for mongo table

    topic.terms
    topicId: 0,
    terms: [
        {"term": "abate", "probability": 0.01},
        ...
    ]

    returns list of dicts
    """

    # normalize and smooth topicterms
    topicterms = np.apply_along_axis(lambda x: x / x.sum(), 1, topicterms)
    docs = []
    for topicid, row in enumerate(topicterms):
        terms = []
        for termindex, probability in enumerate(row):
            term = dictionary.id2token[termindex]
            terms.append({'term': term, 'probability': probability})
        docs.append({'topicId': topicid, 'terms': terms})
    return docs

def create_topics (meta, doctopics, doc_lens, topicterms):
    """
    Create topics data for mongo table

    Create topics
    topicId: 0,
    proportion: 0.0294,
    x: -0.13,
    y: 0.115,
    authors: [...],
    locations: [...],
    keywords: [...],
    publishers: [...],
    topDocs: [1, 5, 345, 657, 34503]

    returns list of dicts
    """

    proportions = compute_proportions(doctopics, doc_lens)
    coordinates = compute_coordinates(topicterms)
    topdocs = compute_top_docs(doctopics)
    subsets = subset_proportions(meta, doctopics, doc_lens)
    authors = subsets[0]
    locations = subsets[1]
    keywords = subsets[2]

    # foreach topic
    docs = []
    for i in range(doctopics.shape[1]):
        docs.append({'topicId': i,
            'proportion': float(proportions[0][i]),
            'x': float(coordinates[i][0]),
            'y': float(coordinates[i][1]),
            'topAuthors':  list(
                authors[i].sort_values(ascending=False)[0:10].index),
            'topLocations': list(
                locations[i].sort_values(ascending=False)[0:10].index),
            'topKeywords': list(
                keywords[i].sort_values(ascending=False)[0:10].index),
            'topDocs': [int(d) for d in topdocs[0:10, i]]})
    return docs


def compute_proportions(doctopics, doc_lens):
    """ Compute corpus wide topic proportions """
    weighted = np.multiply(doctopics.todense(), doc_lens.T)
    return np.array(np.sum(weighted, axis = 0) / np.sum(weighted))

def compute_coordinates(topicterms):
    """ 
    Compute x and y coordinates for topics using multidimensional scaling of 
    topic terms matrix 
    """

    dists = np.zeros(shape=(topicterms.shape[0], topicterms.shape[0]))

    for i in range(topicterms.shape[0]):
        for j in range(i + 1, topicterms.shape[0]):
            dists[i][j] = jensenshannon(topicterms[i], topicterms[j])
    dists = dists + dists.T

    return MDS(n_components=2, 
            dissimilarity = "precomputed").fit_transform(dists)

def compute_top_docs(doctopics):
    """ returns ndarray rows are topic values are doc ids """
    doctopics = doctopics.todense().A
    topdocs = doctopics.argsort(axis=0)[::-1]
    return topdocs


def compute_topic_proportion (group_indices, weighted):
    """
    For the given group indices, compute proportion for each 
    unique entry in the subset (e.g if subset = author, then each entry
    is a unique author

    returns pandas dataframe, rows are unique values, columns are topics,
    values are mean nonzero proportion
    """
    names = list(group_indices.keys())
    res = np.zeros((len(names), weighted.shape[1]))
    i = 0
    for n,indices in group_indices.items():
        res[i] = np.sum(weighted[indices,], axis=0) / np.sum(weighted[indices,])
        i += 1

    return pd.DataFrame(res, index=names)

def subset_proportions(meta, doctopics, doc_lens):
    """  
    for each metadata grouping, compute the mean nonzero topic proportion 
    return ndarray rows are meta fields (author1, author2, loc1, ...),
    cols are topics, values are mean nonzero proportion

    relevant fields are authors, locations, keywords, publishers
    """
    # multiply doc topics by doc lens (element wise)
    weighted = np.multiply(doctopics.todense(), doc_lens.T)

    # get inds for each unique value
    authors_inds = list_group_by(meta["Author"])
    locations_inds = meta.groupby("Location").indices
    keywords_inds = list_group_by(meta["Keywords"])
    #publisher_inds = meta.groupby("Publisher").indices

    # compute mean nonzero proportion foreach subset
    authors = compute_topic_proportion(authors_inds, weighted)
    locations = compute_topic_proportion(locations_inds, weighted)
    keywords = compute_topic_proportion(keywords_inds, weighted)
    #publisher_inds = compute_topic_proportion(publisher_inds, weighted)
    return [authors, locations, keywords]
