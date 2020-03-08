# For each word calculate the position of its first occurence in each document.
# Store the results in a SQL table.

library(RMySQL)
library(stringr)
library(parallel)
library(rjson)
library(tm)


# Constants
dtm_fpath <- "/dsl/eebo/topicmodels/2020.02.19/dtm.rds"
n <- 20  # number of clusters


# Initilize cluster
cluster <- makeCluster(n)
clusterEvalQ(cluster, library("stringr"))
clusterEvalQ(cluster, library("rjson"))


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


# Log a message to stdout
log <- function(msg) {
    print(paste(Sys.time(), msg))
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
process_term <- function(term) {
    res <- lapply(truncated, get_firstpos, term)
    res <- res[res != -1]
    toJSON(res)
}


# Load truncated documents from the database
log("loading truncated documents from database")
query <- "SELECT File_ID, Lemma FROM Truncated_Corpus"
res <- dbSendQuery(con_corpus, query)
docs <- fetch(res, n = -1)
docs$Lemma <- sapply(docs$Lemma, function(x) strsplit(x, "\t")[[1]])

truncated <- docs$Lemma
names(truncated) <- docs$File_ID

# Load document term matrix
dtm <- readRDS(dtm_fpath)
terms <- colnames(dtm)

# Export functions and shared data to cluster
clusterExport(cl = cluster, "get_firstpos")
clusterExport(cl = cluster, "truncated")

log("finding positions")
res <- parLapply(cl = cluster, terms, process_term)

df <- data.frame(do.call(rbind, res))
rownames(df) <- terms

log("writing to db")
dbWriteTable(con_models, "first_pos", df, overwrite = TRUE, append = FALSE,
             row.names = FALSE)

# saveRDS(res, file = "res.rds")
