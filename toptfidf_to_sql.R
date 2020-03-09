library(tm)
library(RMySQL)
library(stringr)
library(parallel)
library(rjson)


# Constants
odir <- "/dsl/eebo/topicmodels/2020.02.19/"
n <- 100  # number of top documents to store in db table
sparsity <- 0.995
nclust <- 20

# Configure connections to the database
con_corpus <- dbConnect(MySQL(),
                        user = "avkoehl",
                        password = "E]h+65S<5t395=!k",
                        dbname = "quintessence_corpus",
                        host = "127.0.0.1")
con_models <- dbConnect(MySQL(),
                        user = "avkoehl",
                        password = "E]h+65S<5t395=!k",
                        dbname = "EEBO_Models",
                        host = "127.0.0.1")

# Initilize cluster
cluster <- makeCluster(nclust)
clusterEvalQ(cluster, library("stringr"))


log <- function(msg) {
    print(paste(Sys.time(), msg))
}


findMostFreqDocs <- function(x, n) {
    docs <- rownames(x)
    f <- factor(x$j, seq_len(x$ncol))
    is <- split(x$i, f)
    vs <- split(x$v, f)
    y <- Map(function(i, v, n) {
             p <- order(v, decreasing = TRUE)[seq_len(n)]
             v <- v[p]
             names(v) <- docs[i[p]]
             v
         },
         is, vs, pmin(lengths(vs), n))
    names(y) <- x$dimnames[[2L]]
    y
}


# Return the position of the first occurence of a term given a vector of words
# If the term is not found return -1
get_firstpos <- function(words, term) {
    pos <- match(1, str_detect(words, term))

    if (is.na(pos))
        pos <- -1

    pos
}


# For a given term return a json object where keys are file id and the values
# are the posisions of the first occurence of the term
process_term <- function(term, truncated, cluster) {
    res <- parLapply(cl = cluster, truncated, get_firstpos, term)
    res <- res[res != -1]
    toJSON(res)
}


log("reading in dtm")
dtm <- readRDS(paste(odir, "dtm.rds", sep = ""))
terms <- colnames(dtm)

# Load truncated documents from the database
log("loading truncated documents from database")
query <- "SELECT File_ID, Lemma FROM Truncated_Corpus"
res <- dbSendQuery(con_corpus, query)
docs <- fetch(res, n = -1)
docs$Lemma <- sapply(docs$Lemma, function(x) strsplit(x, "\t")[[1]])

truncated <- docs$Lemma
names(truncated) <- docs$File_ID

log("finding top tfidf terms")
y <- findMostFreqDocs(dtm, n)
l <- list()
for (i in 1:length(y)) {
    docs <- names(y[[terms[i]]])
    l[[terms[i]]] <- docs
}

# Export functions and shared data to clusters
clusterExport(cl = cluster, "get_firstpos")

res <- list()
for (i in 1:length(terms)) {
    print(paste0("    ", i, " of ", length(terms), " iterations"))
    term <- terms[i]
    res[[i]] <- c(term, process_term(term, truncated[l[[term]]], cluster))
}

df <- as.data.frame(do.call(rbind, res))
colnames(df) <- c("word", "positions")

log("writing to db")
dbWriteTable(con_models, "top_tfidf", df, overwrite = TRUE, append = FALSE,
             row.names = FALSE)
