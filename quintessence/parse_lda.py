import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.manifold import MDS

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

def create_topics (lda):
    pass

def compute_proportions(doctopics, doc_lens):
    """ Compute corpus wide topic proportions """
    weighted = np.multiply(doctopics.todense(), doc_lens)
    return np.sum(weighted, axis = 1) / np.sum(weighted)

def compute_coordinates(topicterms):
    """ Compute x and y coordinates for topics using multidimensional scaling of topic terms matrix """

    distances = np.zeros(shape=(topicterms.shape[0], topicterms.shape[0]))

    for i in range(topicterms.shape[0]):
        for j in range(i + 1, topicterms.shape[0]):
            distances[i][j] = jensenshannon(topicterms[i], topicterms[j])
    distances = distances + distances.T

    return MDS(n_components=2, dissimilarity = "precomputed").fit_transform(distances)

def compute_top_docs (doctopics):
    """ input is sparse matrix of doc topics, returns ndarray rows are topic values are doc ids """
    doctopics = doctopics.todense().A
    topdocs = doctopics.argsort(axis=1)[::-1].T
    return topdocs

def subset_proportions(subsets):
    """  for each metadata grouping, compute the mean nonzero topic proportion 
    This is slightly different then the compute proportions function becuase it doesn't take into account the document lengths, treating each document as equal
    """
    # TODO:
    # get unique values in metacol
    values = metacol.unique_values()

    # get inds for each unique value
    # for each set of inds, get corresponding doctopic rows
    # for those rows compute mean topic proportions (excluding zero's?)
    pass

