def TopicModelPipeline(args):

    # 1. get data
    con = Mongo(args.credentialspath)
    corpus = TMCorpus(con.get_metadata(), con.get_topic_model_data())

    # 2. train
    lda = TopicModel(args.tmodir, args.mallet_path, args.num_topics)
    lda.train(corpus.docs)

    # 3. save to database
    con.write_topic_model_data(lda)

def EmbeddingsPipeline(args):

    # 1. get data
    con = Mongo(args.credentialspath)
    corpus = EmbedCorpus(con.get_metadata(), con.get_topic_model_data())

    # 2. train
    embed = Embeddings(args.embedodir)
    embed.train_subsets(corpus.docs_sentences, corpus.subsets)
    embed.train(corpus.sentences)

    # 3. save to database
    con.write_embeddings_data(embed)