from models.topicmodel import TopicModel
from utils.mongo import Mongo
from utils.nlp import normalize

from joblib import Parallel, delayed

print("initializing classes")
mongo = Mongo("../mongo_credentials.json")
lda = TopicModel(model_path = "../data/topicmodel/", 
        mallet_path= "/usr/local/bin/mallet",
        num_topics = 60)

print("running topic model")
docs = mongo.get_topic_model_data()
docs = docs[0:20]
docs = Parallel(n_jobs=4)(delayed(normalize)(doc) for doc in docs)
lda.train(docs)
