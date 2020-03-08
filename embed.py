import time
import gensim
import multiprocessing
import joblib
from joblib import Parallel, delayed
import pymysql.cursors
import pymysql
import nltk
import string


def get_sentences(doc_content):
    sentences = []
    doc_content = doc_content.replace("\t", " ")
    sents = nltk.sent_tokenize(doc_content)
    return sents


# connect to mysql server
connection = pymysql.connect(user="avkoehl", password="E]h+65S<5t395=!k",
                             db="quintessence_corpus", host="127.0.0.1")

# pull the standardized spelling from rows
with connection.cursor() as cursor:
    sql = "SELECT `Standardized` FROM `Standardized`"
    cursor.execute(sql)
    result = cursor.fetchall()

connection.close()

# get list of list of sentences
stopwords = open("../data/stopwords.txt").read().splitlines()
sentences = []  # list of word lists
doc_sents = Parallel(n_jobs=80)(delayed(get_sentences)(r[0]) for r in result)
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
model = gensim.models.Word2Vec(sentences, sg=1, window=15, size=250, workers=80)
print("processed in ", (time.time() - start) / 60, " minutes")

model.save("/dsl/eebo/wordembeddings/2020.03.02/std-full.model", ignore=[])
model.wv.save_word2vec_format('/dsl/eebo/wordembeddings/2020.03.02/std-full.bin', binary=True)
