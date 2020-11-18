""" Main script that runs LDA, word2vec models and updates the database

    usage: poetry run python main.py
"""
from joblib import Parallel, delayed

from quintessence.mongo import Mongo
from quintessence.topicmodel import TopicModel
from quintessence.nlp import normalize_text

# Connect to Database
print("Connecting to Database")
con = Mongo("./mongo_credentials.json")

# Run Topic Model + Write to DB
print("Running Topic Model")
docs = con.get_topic_model_data()
docs = [normalize_text(doc) for doc in docs]
lda = TopicModel("./data/topicmodel/mallet.model", 
        mallet_path = "/usr/local/bin/mallet",
        num_topics = 10)
lda.train(docs)
# write to db
# free model

# Run Word Embedding + Write to DB
# print("Run Word Embeddings")


