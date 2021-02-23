from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
import pandas as pd

def create_doc_frequencies(corpus):
    c = corpus[ ["raw_word_count", "word_count"] ]

    res = c.to_dict("index")
    docs = []
    for k,v in res.items():
        record = {
                "docId": k,
                "word_count_raw": v["raw_word_count"],
                "word_count_preprocessed": v["word_count"],
                }
        docs.append(record)
    return docs

def create_corpus_frequencies(corpus):
    ndocs_per_year = corpus["Date"].value_counts()
    ndocs_per_decade = corpus["decade"].value_counts()

    ntokens_per_year = corpus.groupby("Date").sum("word_count")
    ntokens_per_decade = corpus.groupby("decade").sum("word_count")

    record = {
            "years": list(ndocs_per_year.index),
            "decades": list(ndocs_per_decade.index),
            "ntokens_year": list(ntokens_per_year["word_count"]),
            "ntokens_decade": list(ntokens_per_decade["word_count"]),
            "ndocs_year":  list(ndocs_per_year),
            "ndocs_decade":  list(ndocs_per_decade)
            }

    return record

def create_term_frequencies(corpus, nterms=200000):
    year_terms = compute_year_term_df(corpus, nterms)
    decade_terms = convert_to_decade_term_df(year_terms)

    docs = []
    for term in year_terms: 
        years = year_terms[term]
        decades = year_terms[term]

        record = {
                "term": term,
                "years_freq": list(years),
                "decades_freq": list(decades)
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

def convert_to_decade_term_df(df):
    df = df.copy()
    df.index = df.index.map(lambda x: x[0:3] + '0')
    return df.groupby("Date").sum()

