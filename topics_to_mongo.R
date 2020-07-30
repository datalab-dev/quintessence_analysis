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
    res <- lapply(res, topn, 5)
    
    sapply(res, toJSON)
}

# read in metadata csv
meta <- read.csv(meta_path, row.names = NULL, stringsAsFactors = FALSE)
qid <- as.data.frame(readRDS(qid_path))
meta <- meta[match(qid$File_ID, meta$File_ID),]
meta$QID <- qid$QID

# read in doc topics
doc_topics <- as.data.frame(readRDS(dt_path))
get_id <- function(x) sub("^([^.]*).*", "\\1", basename(x))
doc_topics$File_ID <- sapply(rownames(doc_topics), get_id)
doc_topics <- doc_topics[match(qid$File_ID, doc_topics$File_ID),]
doc_topics$QID <- qid$QID
doc_topics$File_ID <- NULL

# read in topic terms
topic_terms <- as.data.frame(t(readRDS(tt_path)))

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

# add proportions
topic_freq <- colSums(doc_topics[,1:ntopics], na.rm = TRUE)
topics$proportion <- topic_freq / sum(topic_freq)

# add proportions
df <- data.frame(topics)
colnames(df)[1] <- "_id"

# write to db
m <- mongo("topics", url = mongo_url)
m$remove('{}')
m$insert(df, rownames = FALSE)
