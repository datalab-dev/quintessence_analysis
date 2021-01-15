import re

import nltk
from nltk.corpus import stopwords

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
