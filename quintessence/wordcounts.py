from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from joblib import delayed
from joblib import Parallel
import pandas as pd

from quintessence.nlp import normalize_text

def create_frequencies_datamodel(corpus, workers=4):
    collections = {}

    print("preprocess")
    corpus["raw_word_count"] = corpus["docs"].apply(lambda x: len(x.split()))
    corpus["docs"] = Parallel(n_jobs=workers)(delayed(
        normalize_text(d) for d in corpus["docs"]))
    corpus["Word_count"] = corpus["docs"].apply(len)

    # frequencies.docs
    print("doc frequences")
    collections["frequences.docs"] = create_doc_frequencies(corpus)

    # frequencies.corpus
    print("corpus frequencies")
    collections["frequences.corpus"] = create_corpus_frequencies(corpus)

    # frequencies.terms
    print("term frequencies")
    collections["frequences.terms"] = create_term_frequencies(corpus)

    return collections

def create_doc_frequencies(corpus):
    c = corpus[ ["raw_word_count", "word_count"] ]

    res = c.to_dict("index")
    docs = []
    for k,v in res.items():
        record = {
                "_id": k,
                "word_count_raw": v["raw_word_count"],
                "word_count_preprocessed": v["word_count"],
                }
        docs.append(record)
    return docs

def create_corpus_frequencies(corpus):
    df = corpus[ [ "Date", "word_count"] ]
    groups = df.groupby("Date")

    nd = pd.DataFrame(groups.size(), columns=["doc_count"])
    nt = groups.sum()

    b = pd.merge(nd, nt, on="Date")
    res = b.to_dict()
    
    record = {
            "word_count": res["word_count"],
            "doc_count": res["doc_count"]
            }

    return record

def create_term_frequencies(corpus, nterms=200000):
    year_terms = compute_year_term_df(corpus, nterms)

    docs = []
    for term in year_terms: 
        years = year_terms[term]
        res = years.to_dict()

        record = {
                "_id": term,
                "freq": res,
                }
        docs.append(record)
    return docs

def compute_year_term_df(corpus, nterms):
    years = corpus.groupby("Date")["docs"].sum() # this works concats the lists into one
    dictionary = Dictionary(years)
    dictionary.filter_extremes(no_below = 0,
            no_above=1,keep_n=nterms) # keep only 200k most frequent words
    terms = dictionary.token2id.keys()
    docs = [dictionary.doc2bow(doc) for doc in years]
    dtm = corpus2csc(docs).todense().T
    dtm = pd.DataFrame(index = years.index, data=dtm, columns=terms)
    return dtm
