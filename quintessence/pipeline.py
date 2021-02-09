from quintessence.corpus import TMCorpus
from quintessence.corpus import EmbedCorpus
from quintessence.embeddings import Embeddings
from quintessence.mongo import Mongo
from quintessence.topicmodel import TopicModel

def TopicModelPipeline(args):
    print("Training topic Model")

    # 1. get data
    print("... preprocessing")
    con = Mongo(args["mongo_credentials"])
    corpus = TMCorpus(con.get_metadata(), 
            con.get_topic_model_data(),
            args["ncores"])

    # 2. train
    print("... training")
    lda = TopicModel(args["topic_model"]["tmodir"])
    lda.train(corpus.docs, 
            args["topic_model"]["mallet_path"], 
            args["topic_model"]["num_topics"])

    # 3. save to database
    print("... saving to DB")
    con.write_topic_model_data(lda)

def EmbeddingsPipeline(args):
    print("training word embeddings")

    # 1. get data
    print("... preprocessing")
    con = Mongo(args["mongo_credentials"])
    corpus = EmbedCorpus(con.get_metadata(), 
            con.get_embeddings_data(),
            args["ncores"])

    # 2. train
    print("... training")
    embed = Embeddings(args["embedding"]["embedodir"])
    embed.train_all(corpus.doc_sentences, 
            corpus.subsets,
            args["embedding"]["sg"],
            args["embedding"]["window"],
            args["embedding"]["size"],
            args["ncores"])

    # 3. save to database
    print("... saving to DB")
    con.write_embeddings_data(embed)
