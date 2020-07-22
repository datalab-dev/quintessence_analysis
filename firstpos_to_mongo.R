library(mongolite)
library(jsonlite)
library(stringr)
library(parallel)
library(tm)


mongo_url <- "mongodb://localhost:27017"
dtm_fpath <- "../tmp/dtm.rds"
n <- 20  # number of clusters

cluster <- makeCluster(n)
clusterEvalQ(cluster, library("stringr"))
clusterEvalQ(cluster, library("jsonlite"))


# Log a message to stdout
log <- function(msg) {
    print(paste(Sys.time(), msg))
}


# Return the position of the first occurence of a term given a vector of words
# If the term is not found return -1
get_firstpos <- function(i, term, truncated) {
    words <- truncated[[i]]
    pos <- match(1, str_detect(words, term))
    
    if (is.na(pos))
        pos <- -1
    
    list(
        qid = unbox(as.numeric(names(truncated)[i])),
        position = unbox(pos)
    )
}


# For a given term return a json object where keys are file id and the values
# are the posisions of the first occurence of the term
process_term <- function(term) {
    res <- lapply(1:length(truncated), get_firstpos, term, truncated)
    res <- res[sapply(res, function(x) x$position != -1)]
    data <- list(
        `_id` = unbox(term),
        firstPositions = res
    )
    toJSON(data)
}


# load truncated documents from the database
m <- mongo("docs.truncated", url = mongo_url)
docs <- m$find(
    query = '{}',
    fields = '{"lemma": true}'
)
truncated <- lapply(docs$lemma, function(x) strsplit(x, "\t")[[1]])
names(truncated) <- docs$`_id`

# load document term matrix
dtm <- readRDS(dtm_fpath)
terms <- colnames(dtm)

# find first positions
clusterExport(cl = cluster, "get_firstpos")
clusterExport(cl = cluster, "truncated")
res <- parSapply(cl = cluster, terms, process_term)

# write to db
m <- mongo("terms.positions", url = mongo_url)
m$insert(res)
