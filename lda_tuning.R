library(RMySQL)
library(tm)
library(topicmodels)
library(lda)
library(ldatuning)
library(parallel)

odir = "/dsl/eebo/topicmodels/2020.02.19/"
start = 25
end = 100
interval = 5
ncores = 20
control = list(seed = 77, verbose = 100)

log <- function(msg) {
    print(paste(Sys.time(), msg))
}

# load the dtm
log("loading the dtm")
dtm <- readRDS(paste(odir, "dtm.rds", sep = ""))

# random sample one third of the documents [ndoc = 59635]
doc_subset <- sample(1:nrow(dtm), size = 19878)
dtm_subset <- dtm[doc_subset,]

log("running metrics")
result <- FindTopicsNumber(
    dtm = dtm_subset,
    topics = seq(start, end, interval),
    metrics = c('Griffiths2004', 'CaoJuan2009',
                'Arun2010', 'Deveaud2014'),
    control = control,
    mc.cores = ncores,
    verbose = TRUE
)

log("saving the results")
saveRDS(result, file="../data/ldatuning_results.rds")
