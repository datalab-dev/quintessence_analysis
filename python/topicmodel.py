import subprocess

from utils.mongo import db


class TopicModel:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)


    def train(self):
        """Train topic model using mallet."""
        # TODO load the corpus from mongo

        # TODO write text files to a temp directory

        # TODO import corpus in mallet
        # bin/mallet import-dir \
            # --input /dsl/eebo/2020.02.03-phase1-phase2/text \
            # --output /dsl/eebo/topicmodels/2020.02.19/topic-input.mallet \
            # --remove-stopwords \
            # --extra-stopwords stopwords.txt \
            # --stop-pattern-file regexremove.txt \
            # --keep-sequence

        # TODO prune
        # bin/mallet prune \
        #     --input /dsl/eebo/topicmodels/2020.02.19/topic-input.mallet \
        #     --output /dsl/eebo/topicmodels/2020.02.19/topic-input-pruned.mallet \
        #     --prune-document-freq 298

        # TODO train model
        # bin/mallet train-topics \
        #     --input /dsl/eebo/topicmodels/2020.02.19/topic-input-pruned.mallet \
        #     --num-topics 90 \
        #     --num-iterations 1000 \
        #     --output-model /dsl/eebo/topicmodels/2020.02.19/topic-model-90.bin \
        #     --output-state /dsl/eebo/topicmodels/2020.02.19/topic-state-90.gz \
        #     --output-doc-topics /dsl/eebo/topicmodels/2020.02.19/doctopics-90.dat \
        #     --output-topic-keys /dsl/eebo/topicmodels/2020.02.19/keys-90.dat \
        #     --num-top-words 30

        pass

    def load_model(self):
        """Load model from mallet ouput files."""
        # TODO load dtm

        # TODO load doc topics

        # TODO load topic terms

        pass


    def firstpos(self):
        """
        Create terms.positions
        """"
        # TODO load a list of terms from somewhere

        # load truncated documents from the database
        cursor = db['docs.truncated'].find({}, {'lemma': 1})

        docs = []
        for term in terms:
            tmp = {'_id': term, 'firstPositions': []}
            for doc in cursor:
                pos = doc['lemma'].split('\t').index(term)
                tmp.firstPositions.append({'qid': doc['_id'], 'position': pos})
            docs.append(tmp)

        db['terms.positions'].remove({})
        db['terms.positions'].insert_many(docs)


    def doctopics(self):
        """Create docs.topics""""
        pass


    def topicterms(self):
        """"Create topics.terms"""
        pass


    def topics(self):
        """Create topics""""
        pass


    def update(self):
        self.train()
        self.load_model()
        self.firstpos()
        self.doctopics()
        self.topicterms()
        self.topics()
