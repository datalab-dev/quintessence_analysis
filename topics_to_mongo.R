library(mongolite)
library(jsonlite)
library(dplyr)
library(purrr)


mongo_url <- "mongodb://localhost:27017"
meta_path <- "../tmp/eebo_phase1_2_meta.csv"
tt_path <- "../tmp/topicterms-75.rds"
dt_path <- "../tmp/doctopics-75.rds"
ldavis_path <- "../tmp/ldavis-75.json"
qid_path <- "../tmp/qid.rds"
ntopics <- 75
ntopfields <- 5  # number of top field values listed per topic
ntopdocs <- 5  # number of top docs listed per topic
beta <- 0.01  # topic model param


# Names of n largest values in list
topn <- function(x, n) {
    names(head(sort(unlist(x), decreasing = TRUE), n))
}


# Get the top metadata fields for each topic
get_top_fields <- function(meta, doc_topics, colname) {
    # list of groups
    groups <- meta %>% group_by_at(colname) 
    grouprows <- groups %>% group_rows()
    groupnames <- (groups %>% group_data())[[colname]]
    
    # subset and take mean of nonzero scores
    get_scores <- function(subset) {
        qids <- meta[subset,]$QID
        
        if (length(qids) == 0)
            return(rep(0, ntopics))
        
        data <- doc_topics[doc_topics$QID %in% qids, 1:75]
                
        if (nrow(data) == 0)
            return(rep(0, ntopics))
        
        is.na(data) <- data == 0
        colMeans(data, na.rm = TRUE)
    }
    res <- lapply(grouprows, get_scores)
    names(res) <- groupnames
    
    res <- transpose(res)
    topn <- function(x, n) names(head(sort(unlist(x), decreasing = TRUE), n))
    res <- lapply(res, topn, ntopfields)
    
    # sapply(res, toJSON)
}


# Get top documents for topic
get_top_docs <- function(doc_topics) {
    qids <- doc_topics$QID
    doc_topics$QID <- NULL
    topic_docs <- t(doc_topics)
    colnames(topic_docs) <- qids
    
    topn <- function(x, n) {
        vec <- head(sort(unlist(x), decreasing = TRUE), n)
        as_list <- function(i) list(qid = as.integer(names(vec)[i]), 
                                    score = vec[i])
        lapply(1:length(vec), as_list)
    }
    process_topic <- function(topic) {
       topn(topic_docs[topic,], ntopdocs)
    }
    lapply(1:nrow(topic_docs), process_topic)
}


# Get top terms for topic
get_top_terms <- function(topic_terms) {
    topn <- function(x, n) {
        vec <- head(sort(unlist(x), decreasing = TRUE), n)
        as_list <- function(i) list(term = names(vec)[i], 
                                    score = vec[i])
        lapply(1:length(vec), as_list)
    }
    process_topic <- function(topic) {
        topn(topic_terms[topic,], ntopdocs)
    }
    lapply(1:nrow(topic_terms), process_topic)
}


# read in metadata csv
meta <- read.csv(meta_path, row.names = NULL, stringsAsFactors = FALSE)
qid <- as.data.frame(readRDS(qid_path))
meta <- meta[match(qid$File_ID, meta$File_ID),]
meta$QID <- qid$QID

# read in doc topics
theta <- t(readRDS(dt_path))
alpha <- 5 / nrow(theta) # mallet default is 5/K
theta <- theta + alpha # smooth
doc_topics <- t(theta / rowSums(theta)) # normalize
get_id <- function(x) sub("^([^.]*).*", "\\1", basename(x))
doc_topics <- as.data.frame(doc_topics)
doc_topics$File_ID <- sapply(rownames(doc_topics), get_id)
doc_topics <- doc_topics[match(qid$File_ID, doc_topics$File_ID),]
doc_topics$QID <- qid$QID
doc_topics$File_ID <- NULL

# read in topic terms
phi <- as.data.frame(readRDS(tt_path))
phi <- phi + beta # smooth
topic_terms <- as.data.frame(phi / rowSums(phi))

# read in ldavis
ldavis <- rjson::fromJSON(file = ldavis_path)

# construct topics dataframe
topics <- list()
topics$`_id` <- ldavis$mdsDat$topics

# add pca
topics$x <- ldavis$mdsDat$x
topics$y <- ldavis$mdsDat$y

# add top categories
topics$authors <- get_top_fields(meta, doc_topics, "Author")
topics$locations <- get_top_fields(meta, doc_topics, "Location")
topics$keywords <- get_top_fields(meta, doc_topics, "Keywords")
topics$publishers <- get_top_fields(meta, doc_topics, "Publisher")

# add top docs and terms
topics$topDocs <- get_top_docs(doc_topics)
topics$topTerms <- get_top_terms(topic_terms)

# add proportions
topic_freq <- colSums(doc_topics[,1:ntopics], na.rm = TRUE)
topics$proportion <- topic_freq / sum(topic_freq)

# json
topic_list <- transpose(topics)
docs <- sapply(topic_list, toJSON, auto_unbox = TRUE)


df <- data.frame(topics)
colnames(df)[1] <- "_id"

# write to db
m <- mongo("topics", url = mongo_url)
m$remove('{}')
m$insert(docs, rownames = FALSE)
