library(mongolite)
library(jsonlite)
library(dplyr)
library(purrr)
library(tidyr)


mongo_url <- "mongodb://localhost:27017"
ldavis_path <- "../tmp/ldavis-75.json"

ntopfields <- 5  # number of top field values listed per topic
ntopdocs <- 5  # number of top docs listed per topic
ntopterms <- 20 # number of top terms listed per topic


# Names of n largest values in list
topn <- function(x, n) {
    names(head(sort(unlist(x), decreasing = TRUE), n))
}


# Get the top metadata fields for each topic
get_top_fields <- function(meta, doctopics, colname) {
    ntopics <- ncol(doctopics)
    
    if (colname == "keywords")
        meta <- meta %>% separate_rows("keywords", sep = "--")
    
    # list of groups
    groups <- meta %>% group_by_at(colname) 
    grouprows <- groups %>% group_rows()
    groupnames <- (groups %>% group_data())[[colname]]
    
    # subset and take mean of nonzero scores
    get_scores <- function(subset) {
        if (length(subset) == 0)
            return(rep(0, ntopics))
        
        data <- doctopics[subset, 1:75]
                
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
}


# Get top documents for topic
get_top_docs <- function(doctopics) {
    topicdocs <- t(doctopics)
    colnames(topicdocs) <- rownames(doctopics)
    
    topn <- function(x, n) {
        vec <- head(sort(unlist(x), decreasing = TRUE), n)
        as_list <- function(i) list(qid = as.integer(names(vec)[i]), 
                                    score = vec[i])
        lapply(1:length(vec), as_list)
    }
    process_topic <- function(topic) {
       topn(topicdocs[topic,], ntopdocs)
    }
    lapply(1:nrow(topicdocs), process_topic)
}


# Get top terms for topic
get_top_terms <- function(topicterms) {
    topn <- function(x, n) {
        vec <- head(sort(unlist(x), decreasing = TRUE), n)
        as_list <- function(i) list(term = names(vec)[i], 
                                    score = vec[i])
        lapply(1:length(vec), as_list)
    }
    process_topic <- function(topic) {
        topn(topicterms[topic,], ntopterms)
    }
    lapply(1:nrow(topicterms), process_topic)
}

# read metadata from db collection
m <- mongo('docs.metadata', url = mongo_url)
meta <- m$find(query = '{}', fields = '{}')
meta <- meta[order(meta$`_id`),]

# combine keywords into a string
meta$keywords <- lapply(1:nrow(meta), 
                        function(i) paste(meta$keywords[[i]], collapse = "--"))

# read doc topics from db collection
m <- mongo('docs.topics', url = mongo_url)
tmp <- m$find(query = '{}', fields = '{}')
tmp <- tmp[order(tmp$`_id`),]
doctopics <- lapply(1:nrow(tmp), function(i) tmp[i,]$topics[[1]]$probability)
doctopics <- data.frame(do.call("rbind", doctopics), row.names = tmp$`_id`)

# read in topic terms
m <- mongo('topics.terms', url = mongo_url)
tmp <- m$find(query = '{}', fields = '{}')
tmp <- tmp[order(tmp$`_id`),]
topicterms <- lapply(1:nrow(tmp), function(i) tmp[i,]$terms[[1]]$probability)
topicterms <- data.frame(do.call("rbind", topicterms), row.names = tmp$`_id`)
colnames(topicterms) <- tmp[1,]$terms[[1]]$term

# read in ldavis
ldavis <- rjson::fromJSON(file = ldavis_path)

# construct topics dataframe
topics <- list()
topics$`_id` <- ldavis$mdsDat$topics

# add pca
topics$x <- ldavis$mdsDat$x
topics$y <- ldavis$mdsDat$y

# add top categories
topics$authors <- get_top_fields(meta, doctopics, "author")
topics$locations <- get_top_fields(meta, doctopics, "location")
topics$keywords <- get_top_fields(meta, doctopics, "keywords")
topics$publishers <- get_top_fields(meta, doctopics, "publisher")

# add top docs and terms
topics$topDocs <- get_top_docs(doctopics)
topics$topTerms <- get_top_terms(topicterms)

# add proportions
word_counts <- meta$wordCount
topic_freq <- colSums(doctopics[,1:ntopics] * word_counts, na.rm = TRUE)
topics$proportion <- topic_freq / sum(topic_freq)

# json
topic_list <- transpose(topics)
docs <- sapply(topic_list, toJSON, auto_unbox = TRUE)

# write to db
m <- mongo("topics", url = mongo_url)
m$remove('{}')
m$insert(docs, rownames = FALSE)
