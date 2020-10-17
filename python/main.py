from models.topicmodel import TopicModel
from utils.mongo import Mongo
from utils.nlp import normalize

mongo = Mongo("../mongo_credentials.json")
lda = TopicModel("../data/topicmodel/")

lemma = mongo.get_topic_model_data()
t = [normalize(doc) for doc in lemma]

