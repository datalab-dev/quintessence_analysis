def tokenize(self, doc_content, words_only=False):
    """
    Given standardized text return list of cleaned and tokenized sentences.
    """
    cleaned = []
    doc_content = doc_content.replace("\t", " ")
    sentences = nltk.sent_tokenize(doc_content)

    for s in sentences:
        s = s.lower()
        s = s.replace('|', ' ')
        s = s.translate(str.maketrans('', '', string.punctuation))
        words = [w for w in s.split() if w not in self.stopwords]
        cleaned.append(words)

    if words_only:
        cleaned = list(chain.from_iterable(cleaned))

    return cleaned
