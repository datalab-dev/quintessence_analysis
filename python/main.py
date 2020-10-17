from models.topicmodel import TopicModel
from utils.mongo import Mongo

mongo = Mongo("../mongo_credentials.json")
lda = TopicModel("../data/topicmodel/")

lemma = mongo.get_topic_model_data()

