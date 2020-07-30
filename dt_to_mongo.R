library(mongolite)
library(jsonlite)


mongo_url <- "mongodb://localhost:27017"
dt_path <- "../tmp/doctopics-75.rds"
qid_path <- "../tmp/qid.rds"

# doc topics table
df <- as.data.frame(readRDS(dt_path))
get_id <- function(x) sub("^([^.]*).*", "\\1", basename(x))
df$File_ID <- sapply(rownames(df), get_id)

# qid column
qid <- as.data.frame(readRDS(qid_path))
df <- df[match(qid$File_ID, df$File_ID),]
df$QID <- qid$QID
df$File_ID <- NULL

# transpose
rownames(df) <- df$QID
df$QID <- NULL
df <- t(df)
rownames(df) <- 1:nrow(df) # topics

# initialize a mongodb connection and remove existing docs in collection
m <- mongo("topics.docs", url = mongo_url)
m$remove('{}')

# convert dataframe rows to json strings
row_to_json <- function(i) {
    topic <- rownames(df)[i]
    
    to_list <- function(j) {
        frequency <- df[i, j]
        if (is.na(frequency))
            frequency <- 0
        
        list(qid = unbox(as.integer(colnames(df)[j])), 
             frequency = unbox(as.integer(frequency)))
    }
    docs <- lapply(1:ncol(df), to_list)
    
    data <- list(
        `_id` = unbox(as.integer(topic)),
        docs = docs
    )
    toJSON(data, digits = NA)
}
documents <- sapply(1:nrow(df), row_to_json)

# insert all documents
m$insert(documents)
