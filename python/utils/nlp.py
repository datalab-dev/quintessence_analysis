import string
from itertools import chain

import nltk
from nltk.corpus import stopwords

def preprocess(doc_content, words_only=True):
    """
    Given text from db (tab delimited strings) return list of cleaned and tokenized sentences or words.
    """
    cleaned = []
    doc_content = doc_content.replace("\t", " ")
    sentences = nltk.sent_tokenize(doc_content)

    for s in sentences:
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation))
        words = [w for w in s.split() if w not in stopwords.words('english')]
        cleaned.append(words)

    if words_only:
        cleaned = list(chain.from_iterable(cleaned))

    return cleaned
