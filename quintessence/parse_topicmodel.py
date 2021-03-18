import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.manifold import MDS

from quintessence.nlp import list_group_by


def create_topicmodel_datamodel(doctopics, topicterms, meta, dtm):

    # create Decade column
    meta["Date"] = meta["Date"].astype('str')
    meta["Date"] = meta["Date"].replace( {'nan': '0000'} )
    meta["Date"] = meta["Date"].apply(lambda x: x.split('.')[0])
    meta["Decade"] = meta["Date"].apply(lambda x: x[0:3] + "0")

    collections = {}

    # topics.info
    print("topics.info")
    collections["topics.info"] = create_topics_info(meta, doctopics, dtm)

    # topics.decades
    print("topics.decades")
    collections["topics.decades"] = create_topics_decades_info(meta, doctopics, dtm)

    # topics.proportions
    print("topics.proportions")
    collections["topics.proportions"] = create_topics_proportions(doctopics, dtm, meta)

    # topics.coordinates
    print("topics.coordinates")
    collections["topics.coordinates"] = create_topics_coordinates(topicterms)

    print("topics.toprelevanceterms")
    collections["topics.toprelevanceterms"] = create_topics_toptermsrelevances(topicterms, dtm, doctopics)

    # topics.topterms
    print("topics.topterms")
    collections["topics.topterms"] = create_topics_topterms(topicterms)

    # topics.doctopics
    print("topic.doctopics")
    collections["topics.doctopics"] = create_topics_doctopics(doctopics)

    # topics.termstopics
    print("topics.termstopics")
    collections["topics.termstopics"] = create_topics_termstopics(topicterms)

    return collections

def create_topics_proportions(doctopics, dtm, meta):
    """ 
    create topics.proportions 
    _id: 0
    proportion:
    decades: {
        1470: 0.03,
        1480: 0.04,
    }
    years: {
        1470: 0.03,
        1471: 0.84,
    }
    """
    ntopics = doctopics.shape[1]
    doclens = dtm.sum(axis=1) # row sums
    proportions = compute_proportions(doctopics, doclens)

    decades = compute_topic_proportion(meta.groupby("Decade").groups, doctopics, doclens)
    years = compute_topic_proportion(meta.groupby("Date").groups, doctopics, doclens)

    docs = []
    for i in range(ntopics):
        record = {
                "_id": i,
                "proportion": proportions.loc[str(i)],
                "decades": decades[i].to_dict(),
                "years": years[i].to_dict()
                }
        docs.append(record)
    return docs

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

def create_topics_coordinates(topicterms):
    coordinates = compute_coordinates(topicterms)
    docs = []

    for i in range(coordinates.shape[0]):
        docs.append({'_id': i,
            'x': float(coordinates[i][0]),
            'y': float(coordinates[i][1]),
            })
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

def make_info_record(topic, authors, keywords, locations, doctopics):
    """ creates dictionary. topAuthors, topAuthorsScores, topLocations, 
    topLocationsScores, topKeywords, topKeywordsScores """
    a = authors[topic].sort_values(ascending=False).head(10)
    k = keywords[topic].sort_values(ascending=False).head(10)
    l = locations[topic].sort_values(ascending=False).head(10)
    d = doctopics[str(topic)].sort_values(ascending=False).head(10)

    record = {
            "_id": topic,
            "topAuthors": list(a.index),
            "topAuthorsScores": a.tolist(),
            "topKeywords": list(k.index),
            "topKeywordsScores": k.tolist(),
            "topLocations": list(l.index),
            "topLocationsScores": l.tolist(),
            "topDocs": list(d.index),
            "topDocsScores": d.tolist(),
            }
    return record

def create_topics_info (meta, doctopics, dtm):
    """
    Create topics data for mongo table

    Create topics.info

     _id: 0,
     topAuthors: [...],
     topLocations: [...],
     topKeywords: [...],
     topDocs: [1, 5, 345, 657, 34503]

    returns list of dicts one dict per topic
    """

    ntopics = doctopics.shape[1]
    doclens = dtm.sum(axis=1) # row sums

    # full
    topAuthors = compute_topic_proportion(list_group_by(meta["Author"]),
                                          doctopics, doclens)
    topKeywords = compute_topic_proportion(list_group_by(meta["Keywords"]),
                                          doctopics, doclens)
    topLocations = compute_topic_proportion(meta.groupby("Location").groups,
                                          doctopics, doclens)
    docs = []
    for i in range(ntopics):
        record = make_info_record(
                i, topAuthors, topKeywords, topLocations, doctopics)
        docs.append(record)
    return docs

def create_topics_decades_info(meta, doctopics, dtm):
    """ returns list of dictionaries """
    # _id: 1470
    # topics: {
    #     0: {
    #     topAuthors: [ ],
    #     topLocations: [ ],
    #     topKeywords: [ ],
    #     },
    #    1: {
    #      ... }
    # }

    doclens = dtm.sum(axis=1)
    ntopics = doctopics.shape[1]
     
    decades_dict = []
    decades_inds = meta.groupby("Decade").groups

    docs = []
    for d,inds in decades_inds.items():
        topics = {}
        d_meta = meta.loc[inds]
        d_doctopics = doctopics.loc[inds]
        d_doclens = doclens.loc[inds]

        tda = compute_topic_proportion(list_group_by(d_meta["Author"]), d_doctopics, d_doclens)
        tdk = compute_topic_proportion(list_group_by(d_meta["Keywords"]), d_doctopics, d_doclens)
        tdl = compute_topic_proportion(d_meta.groupby("Location").groups, d_doctopics, d_doclens)

        for i in range(ntopics):
            topics[str(i)] = make_info_record(i, tda, tdk, tdl, doctopics)
            del topics[str(i)]["_id"]
        record = {
                "_id": d,
                "topics": topics
                }
        docs.append(record)
                                            
    return docs

def create_topics_toptermsrelevances(topicterms, dtm, doctopics):
    """ 
      Term relevance is defined as an interpolation between log(P(term|topic)) and
      log(P(term|topic) / P(term)). The interpolation is parameterized with a lambda
      term between 0 and 1. We compute:

      relevance(word, topic | lambda) =
           lambda * log(P(term|topic)) + (1 - lambda) * log(lift(term, topic))
      where lift(term, topic) = P(term|topic) / P(term).

    """

    # add smoothing to topicterms
    # default in mallet is 0.01
    # https://stackoverflow.com/a/44591188
    tt = topicterms + (0.01 / topicterms.shape[1])
    tt = tt.div(tt.sum(axis=1), axis=0)

    # topic freq
    topic_freq = doctopics.multiply(dtm.sum(axis=1), axis=0).sum(axis=0)
    topic_freq.index = topic_freq.index.astype(int)

    # topic proportion
    proportion  = compute_proportions(doctopics, dtm.sum(axis=1))

    # term topic freq (an estimate, could use model.wordstopics but its not the same... better would be an average over iterations after stable)
    term_topic_freq = tt.multiply(topic_freq.values, axis=0)

    # term freq
    # note its NOT: tf = dtm.sum(axis=0) because won't match; same workaround as ldavis uses in source code...
    tf = term_topic_freq.sum(axis=0)

    #  term count / total count 
    p_term = tf / tf.sum()
 
    docs = []
    for topic in range(topicterms.shape[0]):
        topic_term_ranks = {}
        p_term_topic = tt.loc[topic]
        lift = np.divide(p_term_topic, p_term)

        for step,lam in enumerate(np.arange(0, 1.1, step=0.1)):
            lam = round(lam, 1)
            term_topic_relevance = lam * np.log(p_term_topic) + (1 - lam) * np.log(lift)
            topterms = term_topic_relevance.sort_values(ascending=False).head(30)
            topic_term_ranks[str(step)] = {
               "lambda": lam,
               "terms" : topterms.index.to_list(),
               "relevances" : topterms.to_list(),
               "estimatedTermFreqTopic" : term_topic_freq.loc[topic][topterms.index].to_list(),
               "overallFreq" : tf.loc[topterms.index].to_list()
            }

        record = {
                "_id": topic,
                "topterms": topic_term_ranks
                }
        docs.append(record)
 
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

def compute_topic_proportion (group_indices, doctopics, doc_lens, mincount = 3):
    """
    For the given group indices, compute proportion for each 
    unique entry in the subset (e.g if subset = author, then each entry
    is a unique author

    returns pandas dataframe, rows are unique values, columns are topics,
    values are mean nonzero proportion
    """

    # remove indices if less than 'mincount' documents associated
    toremove = []
    for k,v in group_indices.items():
        if len(v) < mincount: toremove.append(k)
    for k in toremove:
        del group_indices[k]

    names = list(group_indices.keys())
    res = np.zeros((len(names), doctopics.shape[1]))

    i = 0
    for n,indices in group_indices.items():
        dt = doctopics.loc[indices]
        dl = doc_lens[indices]
        res[i] = compute_proportions(dt, dl)
        i += 1

    return pd.DataFrame(res, index=names)
