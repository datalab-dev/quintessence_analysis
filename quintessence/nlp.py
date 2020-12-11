import re

import numpy as np
import nltk
from nltk.corpus import stopwords
from scipy.spatial.distance import jensenshannon
from sklearn.manifold import MDS

def normalize_text(text,
        lower = True,
        punct = True,
        digits = True,
        stopwords = stopwords.words('english'),
        minlen = 2,
        ):
    """
    Given string of words (whitespace delim), return string of normalized words
    """
    PUNCT_RE = r'[\[\]|!"#$%&\'()*+,./:;<=>?@\^_`{|}~]'

    if lower: text = text.lower()
    if punct: text = re.sub(PUNCT_RE, '', text)
    if digits: text = re.sub(r'\d+', '', text)
    text = ' '.join(x for x in text.split() if x not in stopwords and len(x) > minlen)

    # handle excess whitespace from removed terms
    text = " ".join(text.split())
    return text

def compute_proportions(doctopics, doc_lens):
    """ Compute corpus wide topic proportions """
    weighted = np.multiply(doctopics.todense(), doc_lens)
    return np.sum(weighted, axis = 1) / np.sum(weighted).A

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

def get_subsets (metacol):
    """   """


def group_by_proportions(doctopics, metacol): 
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

