from quintessence.embeddings import Embeddings
from quintessence.mongo import Mongo
from quintessence.topicmodel import TopicModel

def TopicModelPipeline(args):
    print("Training topic Model")

    # 1. get data
    print("Get data")
    con = Mongo(args["mongo_credentials"])
    corpus = con.get_topic_model_data()

    # 2. train
    print("Run Topic Model")
    lda = TopicModel(args["topic_model"]["tmodir"])
    lda.train(corpus,
            args["topic_model"]["mallet_path"], 
            args["topic_model"]["num_topics"],
            workers = args["ncores"])

    # 3. save to database
    print("Saving to DB")
    con.write_topic_model_data(corpus, lda)

def EmbeddingsPipeline(args):
    print("training word embeddings")

    # 1. get data
    print("Get data")
    con = Mongo(args["mongo_credentials"])
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
