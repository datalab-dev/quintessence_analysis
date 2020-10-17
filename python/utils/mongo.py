import json

from pymongo import MongoClient
from gensim.models.wrappers import LdaMallet

class Mongo:
    def __init__(self, credentials_path):
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
            url = f"mongodb://{credentials['host']}:{credentials['port']}"
    
        # try to form client connection
        try:
            client = MongoClient(url)
        except pymongo.errors.ConnectionFailure:
             print(f"Failed to connect to {url}")
    
        self.db = client[credentials['database']]

    def get_topic_model_data(self):
        res = list(self.db["docs.lemma"].find({}))
        docs = [r["lemma"] for r in res]
        return docs

#    def write_topic_model_data(model):
#        doctopics = model.load_document_topics()
#        phi = self.model.load_word_topics()
#        beta = 0.01 # mallet default
#        phi = phi + beta
#
#        # Create docs.topics
#        docs = []
#        for qid, row in enumerate(doc_topics):
#            topics = []
#            for topic, probability in row:
#                topics.append({'topicId': topic, 'probability': probability})
#            docs.append({'_id': qid, 'topics': topics})
#
#        db['docs.topics'].remove({})
#        db['docs.topics'].insert_many(docs)
#
#       # Create terms.topics
#        termtopics = np.apply_along_axis(lambda x: x / x.sum(), 0, phi) # normalize and smooth document topics
#        topicterms = np.transpose(termtopics)
#
#        docs = []
#        for i in range(topicterms.shape[1]):
#            term = self.mode.id2word[i]
#            topics = [{'topicId': j+1, 'probability': p} for j, p in
#                      enumerate(topicterms[term])]
#            docs.append({'_id': term, 'topics': topics})
#
#        db['terms.topics'].remove({})
#        db['terms.topics'].insert_many(docs)
#
#        """
#        Create topics
#        """
#        # TODO everything (see topics_to_mongo.R)
#        pass
