import json

from quintessence.corpus import TMCorpus
from quintessence.corpus import EmbedCorpus
from quintessence.embeddings import Embeddings
from quintessence.mongo import Mongo
from quintessence.topicmodel import TopicModel

args = json.load(open('config.json'))

def TopicModelPipeline(args):

    # 1. get data
    con = Mongo(args["mongo_credentials"])
    corpus = TMCorpus(con.get_metadata(), con.get_topic_model_data())

    # 2. train
    lda = TopicModel(args["topic_model"]["tmodir"])
    lda.train(corpus.docs, 
            args["topic_model"]["mallet_path"], 
            args["topic_model"]["num_topics"])

    # 3. save to database
    con.write_topic_model_data(lda)

def EmbeddingsPipeline(args):

    # 1. get data
    con = Mongo(args["mongo_credentials"])
    corpus = EmbedCorpus(con.get_metadata(), con.get_embeddings_data())

    # 2. train
    embed = Embeddings(args["embedding"]["embedodir"])
    embed.train_all(corpus.doc_sentences, 
            corpus.subsets,
            args["embedding"]["sg"],
            args["embedding"]["window"],
            args["embedding"]["size"],
            args["embedding"]["workers"])

    # 3. save to database
    con.write_embeddings_data(embed)
