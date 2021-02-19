""" 
Main script that runs LDA, word2vec models and updates the database

usage: poetry run python main.py
"""
import json

import numpy as np
import pandas as pd

from quintessence.embeddings import Embeddings
from quintessence.mongo import Mongo
from quintessence.topicmodel import TopicModel

## load in json config file
args = json.load(open("config.json"))

# TOPIC MODEL
con = Mongo(args["mongo_credentials"])
corpus = con.get_topic_model_data()

lda = TopicModel(args["topic_model"]["tmodir"])
lda.train(corpus,
        args["topic_model"]["mallet_path"],
        args["topic_model"]["num_topics"],
        workers = args["ncores"])

con.write_topic_model_data(corpus,lda)


# EMBEDDING
corpus = con.get_embeddings_data()

# 2. train
print("Run Word Embeddings")
embed = Embeddings(args["embedding"]["embedodir"])
embed.train_all(corpus,
       sg = args["embedding"]["sg"],
       window = args["embedding"]["window"],
       size = args["embedding"]["size"],
       workers = args["ncores"])

# 3. save to database
print("Saving to DB")
con.write_embeddings_data(embed)
