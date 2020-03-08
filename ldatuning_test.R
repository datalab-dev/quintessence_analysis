library(RMySQL)
library(tm)
library(topicmodels)
library(lda)
library(ldatuning)
library(parallel)

odir <- "/dsl/eebo/topicmodels/2020.02.19/"

log = function(msg) {
    print(paste(Sys.time(), msg))
}

# load the dtm
log("loading the dtm")
dtm <- readRDS(paste(odir, "dtm.rds", sep = ""))

# random sample one third of the documents [ndoc = 59635]
doc_subset <- sample(1:nrow(dtm), size = 19878)
dtm_subset <- dtm[doc_subset,]

# lda config
ldaformat <- dtm2ldaformat(dtm)
ldaformat_subset <- dtm2ldaformat(dtm_subset)

# lda FULL
print("lda FULL")
ptm <- proc.time()
model <- lda.collapsed.gibbs.sampler(documents = ldaformat$documents,
                                     K = 100,
                                     vocab = ldaformat$vocab,
                                     num.iterations = 1,
                                     alpha = 1,
                                     eta = 0.1,
                                     burnin = 0,
                                     trace = 1L)
proc.time() - ptm

# lda SUBSET
print("lda SUBSET")
model_subset <- lda.collapsed.gibbs.sampler(documents = ldaformat_subset$documents,
                                            K = 100,
                                            vocab = ldaformat_subset$vocab,
                                            num.iterations = 1,
                                            alpha = 1,
                                            eta = 0.1,
                                            burnin = 0,
                                            trace = 1L)
proc.time() - ptm

# topicmodels config
control = list(seed = 77, verbose = 1, iter = 1)
method = "Gibbs"

# topicmodels FULL
print("topicmodels FULL")
ptm <- proc.time()
topicmodels::LDA(dtm, k = 100, method = method, control = control)
proc.time() - ptm

# topicmodels SUBSET
print("topicmodels SUBSET")
ptm <- proc.time()
topicmodels::LDA(dtm_subset, k = 100, method = method, control = control)
proc.time() - ptm
