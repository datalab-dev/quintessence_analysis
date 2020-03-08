import csv
import time
import gensim
import multiprocessing
import joblib
from joblib import Parallel, delayed
import nltk
import string
import pymysql.cursors
import pymysql
import re
import pandas as pd


def get_sents(doc_content):
    sentences = []
    doc_content = doc_content.replace("\t", " ")
    sents = nltk.sent_tokenize(doc_content)
    return sents


def train_model(texts, stopwords_fname, odir, name):
    """Given a subset of texts and a filename train and save a word2vec
       model."""
    stopwords = open(stopwords_fname).read().splitlines()

    # get list of list of sentences
    sentences = []  # list of word lists
    doc_sents = Parallel(n_jobs=80)(delayed(get_sents)(t) for t in texts)
    for ds in doc_sents:
        for s in ds:
            s = s.lower()
            s = s.replace('|', ' ')
            s = s.translate(str.maketrans('', '', string.punctuation))
            words = [w for w in s.split() if w not in stopwords]
            sentences.append(words)

    # train model
    start = time.time()
    print("training full model")
    model = gensim.models.Word2Vec(sentences, sg=1, window=15, size=250,
                                   workers=80)
    print("processed in ", (time.time() - start) / 60, " minutes")

    # save models
    model.save(odir + name + '.model', ignore=[])
    model.wv.save_word2vec_format(odir + name + '.txt', binary=False)


def get_corpus_df(meta_fname):
    """Given metadata file, get texts from database and combine into
       dataframe with text, author, location, and decade."""

    # connect to mysql server
    connection = pymysql.connect(user='avkoehl',
                                 password='E]h+65S<5t395=!k',
                                 db='quintessence',
                                 host='127.0.0.1')

    # pull the standardized spelling from rows
    with connection.cursor() as cursor:
        sql = "SELECT File_ID,Standardized FROM EEBO_Corpus"
        cursor.execute(sql)
        result = cursor.fetchall()

    # store docs as a dictionary
    docs = {item[0]: list(item[1:]) for item in result}

    # get metadata and add to dictionary
    with open(meta_fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_num = 0
        for row in csv_reader:
            id, author, location, year = row[1], row[9], row[12], row[13]
            if line_num > 0:
                if not year.isdigit() or int(year) < 1470 or int(year) > 1700:
                    year = ''
                else:
                    year = year[:3]  # truncate to get decade
                docs[id].extend([author, location, year])
            line_num += 1

    df = pd.DataFrame.from_dict(docs,
                                orient='index',
                                columns=['text', 'author', 'location',
                                         'decade'])
    return(df)


def main():
    # set parameters
    meta = '../data/eebo_meta.csv'
    stopwords = '../data/stopwords.txt'
    loc = '../data/locations.txt'  # list of top locations
    auth = '../data/authors.txt'  # list of top authors

    # fields = ['decade', 'location', 'author']
    fields = ['location', 'author']
    odir = '/dsl/eebo/wordembeddings/2020.03.02'

    df = get_corpus_df(meta)

    for field in fields:
        dir = odir + '-' + field + 's/'
        print("training model for " + field + " and outputting in " + odir)

        # remove unwanted rows
        df = df[df[field] != '']  # remove rows where separating field is blank

        if field == 'location':
            locations = open(loc).read().splitlines()
            df = df[df[field].isin(locations)]

        if field == 'author':
            authors = open(auth).read().splitlines()
            df = df[df[field].isin(authors)]

        grouped = df.groupby(field)

        for name, group in grouped:
            texts = group['text'].tolist()
            train_model(texts, stopwords, dir, name)


if __name__ == '__main__':
    main()
