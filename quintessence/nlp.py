import re

import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim

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
    text = ' '.join(
            x for x in text.split() if x not in stopwords and len(x) > minlen)

    # handle excess whitespace from removed terms
    text = " ".join(text.split())
    return text

def sentence_tokenize(text):
    """ Split a text string into a list of sentences (list of words) """
    sentences = nltk.sent_tokenize(text)
    return sentences

def list_group_by(series):
    """ behaves like df.groupby("Date").indices but works on list """
    values = series.explode()
    inds = {}
    for index,k in values.items():
        if k in inds.keys(): inds[k].append(index)
        else: inds[k] = [index]

    # convert to ndarrays to match groupby return 
    for k,v in inds.items():
        inds[k] = np.array(v)
    return inds

def align_vocab(base, other):
    """ given base model and model to be aligned to base, 
    find the overlapping vocab terms, reorder the syn0norm and syn0 matrices
    of the two models to only have the overlapping terms.  
    This is necessary for procrustes alignment to work.
    """

    common_vocab = list(set(base.wv.vocab.keys()) & set(other.wv.vocab.keys()))

    # sort vocab based on combined count
    common_vocab.sort(key=lambda w: base.wv.vocab[w].count + other.wv.vocab[w].count,reverse=True)

    # generate new syn0norm + syn0, vocab dictionary + index2word for each model
    for model in [base, other]:
        indices = [model.wv.vocab[w].index for w in common_vocab]
        model.wv.syn0norm = np.array([model.wv.syn0norm[i] for i in indices])
        model.wv.syn0 = model.wv.syn0norm

        model.index2word = common_vocab
        for i,term in enumerate(common_vocab):
            m.wv.vocab[term] = gensim.models.word2vec.Vocab(index=i, 
                    count=m.wv.vocab[term].count)

    return base,other

def procrustes_alignment(base, other):
    """  Align two models using procrustes alignment. Thanks 'histwords'
    https://github.com/williamleif/histwords/blob/master/vecanalysis/alignment.py
    """
    base, other = align_vocab(base, other)
    m = other.wv.syn0norm.T.dot(base.wv.syn0norm)
    u, _, v = np.linalg.svd(m)
    ortho = u.dot(v)
    other.wv.syn0norm = other.wv.syn0norm.dot(ortho)
    other.wv.syn0 = other.wv.syn0norm
    return base,other
