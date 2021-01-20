""" 
Main script that runs LDA, word2vec models and updates the database

usage: poetry run python main.py
"""
from quintessence.corpus import Corpus
from quintessence.mongo import Mongo
from quintessence.topicmodel import TopicModel
from quintessence.embeddings import Embeddings

def run_topic_model(con):
    """ Given connection to database, run topic model """
    corpus = Corpus(con.get_metadata(), con.get_topic_model_data())
    corpus.topic_model_preprocessing()
    lda = TopicModel("./data/topicmodel/", 
             mallet_path = "/usr/local/bin/mallet",
             num_topics = 5)
    lda.train(corpus.docs)
    con.write_topic_model_data(lda)
    return

# Connect to Database
print("Connecting to Database")
con = Mongo("./mongo_credentials.json")

# Topic Model
# run_topic_model(con)

# Run Word Embedding + Write to DB
corpus = Corpus(con.get_metadata(), con.get_embeddings_data())
corpus.embed_preprocessing()
embed = Embeddings("./data/embeddings/")
embed.train(corpus.sentences)
#con.write_embeddings_data(embed)
