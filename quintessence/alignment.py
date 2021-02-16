import gensim
import numpy as np

def get_common_vocab(m1, m2):
    common_vocab = list(set(m1.wv.vocab.keys()) & set(m2.wv.vocab.keys()))
    common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + 
            m2.wv.vocab[w].count,reverse=True)
    return common_vocab

def update_model_vocab(model, common_vocab):
    """ 
    given model and terms update the vectors + vectors_norm and
    index2word + vocab properties 
    """


    model.wv.init_sims() # otherwise model.wv.vectors can be empty
    inds = [model.wv.vocab[w].index for w in common_vocab]
    model.wv.vectors_norm = np.array( [model.wv.vectors_norm[i] for i in inds] )
    model.wv.vectors = model.wv.vectors_norm

    model.index2word = common_vocab
    updated_vocab = {}
    for i,term in enumerate(common_vocab):
        count = model.wv.vocab[term].count
        updated_vocab[term] = gensim.models.word2vec.Vocab(index = i,
                count = count)
    model.wv.vocab = updated_vocab
    return model

def align_vocab(base, other):
    """ given base model and model to be aligned to base, 
    find the overlapping vocab terms, reorder the syn0norm and syn0 matrices
    of the two models to only have the overlapping terms.  
    This is necessary for procrustes alignment to work.
    """
    common_vocab = get_common_vocab(base,other)

    # no common vocab!
    if len(common_vocab) == 0:
        return None, None

    base = update_model_vocab(base, common_vocab)
    other = update_model_vocab(other, common_vocab)

    return base,other

def procrustes_alignment(base, other):
    """  Align two models using procrustes alignment. Thanks 'histwords'
    https://github.com/williamleif/histwords/blob/master/vecanalysis/alignment.py
    """
    base, other = align_vocab(base, other)

    # failed to find common vocab, can't align!
    if not base:
        return None,None

    m = other.wv.syn0norm.T.dot(base.wv.syn0norm)
    u, _, v = np.linalg.svd(m)
    ortho = u.dot(v)
    other.wv.syn0norm = other.wv.syn0norm.dot(ortho)
    other.wv.syn0 = other.wv.syn0norm
    return base,other
