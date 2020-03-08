library(RMySQL)
library(tm)
library(topicmodels) #for dtm2ldaformat
library(lda)
library(LDAvis)
library(parallel)


# config
K = 60
alpha = K / 100
odir = "/dsl/eebo/topicmodels/2019.10.09-60/"
ncores = 20 # for cleaning

log = function(msg) {
    print(paste(Sys.time(), msg))
}

# load dtm
log("read dtm")
dtm <- readRDS(paste(odir, "dtm.rds", sep = ""))
ldaformat = dtm2ldaformat(dtm)
documents = ldaformat$documents
vocab = ldaformat$vocab

# run the topic model
log("running the topic model")
model = lda.collapsed.gibbs.sampler(
        documents = ldaformat$documents,
        K = K,
        vocab = ldaformat$vocab,
        num.iterations = 1580,
        alpha = alpha,
        eta = 0.1,
        burnin = 0,
        trace = 1L)


dt = t(apply(model$document_sums + 0.5, 2, function(x) x/sum(x)))
tt = t(apply(t(model$topics) + 0.1, 2, function(x) x/sum(x)))
doclens = row_sums(dtm)
wordfreqs = col_sums(dtm)

# save all outputs - dtm, dt, tt, model
log("saving outputs")
# saveRDS(dtm, file=paste(odir, "dtm.rds", sep = ""))
saveRDS(model, file=paste(odir, "model.rds", sep = ""))
saveRDS(dt, file=paste(odir, "doc_topics.rds", sep = ""))
saveRDS(tt, file=paste(odir, "topic_terms.rds", sep = ""))
saveRDS(doclens, file=paste(odir, "doc_lens.rds", sep = ""))
saveRDS(wordfreqs, file=paste(odir, "word_freqs.rds", sep = ""))

# createJSON for ldavis
log("creating json")
ldavis = createJSON(tt, dt, doclens, ldaformat$vocab, wordfreqs,
                    reorder.topics=FALSE)
cat(ldavis, file=paste(odir, "ldavis.json", sep = ""))
