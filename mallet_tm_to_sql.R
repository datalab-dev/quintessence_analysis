library(RMySQL)
library(dplyr)
library(rjson)
library(purrr)

# File paths
tt_path <- "/dsl/eebo/topicmodels/2020.02.19/topicterms-75.rds"
dt_path <- "/dsl/eebo/topicmodels/2020.02.19/doctopics-75.rds"
ldavis_path <- "/dsl/eebo/topicmodels/2020.02.19/ldavis-75.json"
qid_path <- "../quintessence_corpus/qid.rds"
ntopics <- 75


# Configure database connections
modelscon <- dbConnect(MySQL(),
                       user = "avkoehl",
                       password = "E]h+65S<5t395=!k",
                       dbname = "EEBO_Models",
                       host = "127.0.0.1")
corpuscon <- dbConnect(MySQL(),
                       user = "avkoehl",
                       password = "E]h+65S<5t395=!k",
                       dbname = "quintessence",
                       host = "127.0.0.1")


# Get the top metadata fields for each topic
get_top_fields <- function(table, colname) {
    doclist <- get_doclist(table, colname)
    res <- list()
    for (item in doclist) {
        # Subset and take mean of nonzero scores
        get_scores <- function() {
            if (length(item$qids) == 0)
                return(rep(0, ntopics))

            data <- doc_topics[doc_topics$row_names %in% item$qids, 2:76]

            if (length(rownames(data)) == 0)
                return(rep(0, ntopics))

            is.na(data) <- data == 0
            colMeans(data, na.rm = TRUE)
        }
        res[[item$field]] <- get_scores()
    }

    res <- transpose(res)
    topn <- function(x, n) names(head(sort(unlist(x), decreasing = TRUE), n))
    res <- lapply(res, topn, 5)

    sapply(res, toJSON)
}


# For a given field, subset documents by qid and return as list
get_doclist <- function(table, colname) {
    sql <- sprintf("SELECT DISTINCT %s FROM %s;", colname, table)
    fields <- dbGetQuery(corpuscon, sql)
    fields <- unlist(fields)

    res <- list()
    i <- 1
    for (field in fields) {
       sql <- sprintf("SELECT QID FROM %s WHERE %s='%s'", table, colname, field)

       retlist <- tryCatch(
           dbGetQuery(corpuscon, sql),
           error = function(e) e
       )

       if(inherits(retlist, "error"))
           next

       retlist <- dbGetQuery(corpuscon, sql)
       res[[i]] <- list()
       res[[i]]$field <- field
       res[[i]]$qids <- unlist(retlist)
       i <- i + 1
    }

    res
}


# Read in data
topic_terms <- as.data.frame(t(readRDS(tt_path)))
doc_topics <- as.data.frame(readRDS(dt_path))
ldavis <- fromJSON(file = ldavis_path)
qid <- as.data.frame(readRDS(qid_path))

# Construct doc topics table
get_basename <- function(x) strsplit(x, "\\.|\\/")[[1]][8]
fileids <- sapply(rownames(doc_topics), get_basename)
fileid_to_qid <- function(fileid) qid[qid$File_ID == fileid,]$QID
qids <- sapply(fileids, fileid_to_qid)
rownames(doc_topics) <- qids
doc_topics <- tibble::rownames_to_column(doc_topics, "row_names")
doc_topics$File_ID <- fileids

# Construct topic terms table
topic_terms <- tibble::rownames_to_column(topic_terms, "word")

# Construct topic terms pca table
topic_terms_pca <- list()
topic_terms_pca$x <- ldavis$mdsDat$x
topic_terms_pca$y <- ldavis$mdsDat$y
topic_terms_pca$topics <- ldavis$mdsDat$topics
topic_terms_pca$top_authors <- get_top_fields("Authors", "Author")
print("finished authors")
topic_terms_pca$top_locations <- get_top_fields("Metadata", "Location")
print("finished locations")
topic_terms_pca$top_keywords <- get_top_fields("Keywords", "Keyword")
print("finished keywords")
topic_terms_pca$top_publishers <- get_top_fields("Metadata", "Publisher")
print("finished publishers")
topic_terms_pca <- data.frame(topic_terms_pca)

# Construct topic proportions table
topic_proportions <- list()
topic_proportions$topic <- paste0("V", 1:ntopics)
topic_freq <- colSums(doc_topics)
topic_proportions$proportion <- topic_freq / sum(topic_freq)
topic_proportions <- as.data.frame(topic_proportions)

# Write tables to db
dbWriteTable(modelscon, "topic_terms", topic_terms, overwrite = TRUE,
             append = FALSE, row.names = FALSE)
print("wrote topic terms")
dbWriteTable(modelscon, "doc_topics", doc_topics, overwrite = TRUE,
             append = FALSE, row.names = FALSE)
print("wrote doc topics")
dbWriteTable(modelscon, "topic_terms_pca", topic_terms_pca, overwrite = TRUE,
             append = FALSE, row.names = FALSE)
print("wrote topic terms pca")
dbWriteTable(modelscon, "topic_proportions", topic_proportions, overwrite = TRUE,
             append = FALSE, row.names = FALSE)
print("wrote topic proportions")
