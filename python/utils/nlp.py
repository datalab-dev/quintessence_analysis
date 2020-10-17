import string
import re

import nltk
from nltk.corpus import stopwords


def normalize(text):
    """
    Given string of words (whitespace delim), return string of normalized words
    """
    PUNCT_RE = r'[!"#$%&\'()*+,./:;<=>?@\^_`{|}~]'
    s = stopwords.words('english')
    cleaned = []

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(PUNCT_RE, '', text)
    text = re.sub(r'\s\s+', ' ', text)  # Handle excess whitespace
    text = ' '.join(x for x in text.split() if x not in s)
    text = text.strip()  # No whitespace at start and end of string

    return text
