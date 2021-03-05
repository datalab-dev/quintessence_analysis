import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.manifold import MDS

from quintessence.nlp import list_group_by


def create_topicmodel_datamodel(doctopics, topicterms, meta, dtm):
    collections = {}

    # topics
    collections["topics"] = create_topics(meta, doctopics, dtm, topicterms)

    # topics.topterms
    collections["topics.topterms"] = create_topics_topterms(topicterms)

    # topics.doctopics
    collections["topics.doctopics"] = create_topics_doctopics(doctopics)

    # topics.termstopics
    collections["topics.termstopics"] = create_topics_termstopics(topicterms)

    return collections

def create_topics_topterms(topicterms, nterms=100):
    """ 
    create topics.topterms
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
                "terms": list(topicterms.columns[row][0:nterms]),
                "scores": list(tt[i][row][0:nterms])
                }
        docs.append(record)
    return docs

def create_topics_doctopics (doctopics):
    """
    Create topics.doctopics data for mongo table

    topics.doctopics
    _id: 0, (document id)
    topic_distribution: [0.04 ...]

    returns list of dicts
    """
    docs = []
    for i,topics in enumerate(doctopics.to_records(index=False)):
        record = {
                "_id": i,
                "topic_distribution": [t.item() for t in topics] # convert numpy int64 to int
                }
        docs.append(record)
    return docs

def create_topics_termstopics(topicterms):
    """
    Create topics.termstopics data for mongo table

    topics.
    _id: "abate",
    topic_scores: [0.01, ...]

    returns list of dicts
    """
    docs = []
    for term in topicterms:
        record = {
                "_id": term,
                "topic_scores": list(topicterms[term])
                }
        docs.append(record)
    return docs

def create_topics (meta, doctopics, dtm, topicterms):
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

    meta["Date"] = meta["Date"].astype('str')
    meta["Date"] = meta["Date"].apply(lambda x: x.split('.')[0])
    subsets = subset_proportions(meta, doctopics, doc_lens)
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

def subset_proportions(meta, doctopics, doc_lens):
    """  
    for each metadata grouping (e.g London, 'John Donne etc) 
    compute topic prorportions

    return list of dataframes
    """
    # get inds for each unique value
    authors_inds = list_group_by(meta["Author"])
    locations_inds = meta.groupby("Location").groups
    keywords_inds = list_group_by(meta["Keywords"])
    dates_inds = meta.groupby("Date").groups

    # compute mean nonzero proportion foreach subset
    authors = compute_topic_proportion(authors_inds, doctopics, doc_lens)
    locations = compute_topic_proportion(locations_inds, doctopics, doc_lens)
    keywords = compute_topic_proportion(keywords_inds, doctopics, doc_lens)
    dates = compute_topic_proportion(dates_inds, doctopics, doc_lens)
    return [authors, locations, keywords, dates]
