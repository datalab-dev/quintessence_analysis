from joblib import Parallel, delayed

from quintessence.mongo import Mongo
from quintessence.topicmodel import TopicModel
from quintessence.nlp import normalize_text

# Connect to Database
print("Connecting to Database")
con = Mongo("./mongo_credentials.json")

# Run Topic Model
print("Running Topic Model")
docs = con.get_topic_model_data()
docs = [normalize_text(doc) for doc in docs]
lda = TopicModel("./data/topicmodel/mallet.model", 
        mallet_path = "/usr/local/bin/mallet",
        num_topics = 10)
lda.train(docs)

# Run Word Embedding
# print("Run Word Embeddings")

# Write to Database
# print("Writing Results to database")

