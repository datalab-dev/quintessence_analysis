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

# initialize a mongodb connection and remove existing docs in collection
m <- mongo("docs.topics", url = mongo_url)
m$remove('{}')

# convert dataframe rows to json strings
row_to_json <- function(row) {
    to_list <- function(x) list(topicId = unbox(x), 
                                frequency = unbox(row[[x]]))
    topics <- lapply(1:(length(row)-1), to_list)
    data <- list(
        `_id` = unbox(as.numeric(row[[length(row)]])),
        topics = topics
    )
    toJSON(data, digits = NA)
}
docs <- apply(df[1:5,], 1, row_to_json)

# insert all documents
# m$insert(docs)
