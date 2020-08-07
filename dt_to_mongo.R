library(mongolite)
library(jsonlite)


mongo_url <- "mongodb://localhost:27017"
dt_path <- "../tmp/doctopics-75.rds"
qid_path <- "../tmp/qid.rds"


# normalize and smooth document topics
theta <- t(readRDS(dt_path))
alpha <- 5 / nrow(theta) # mallet default is 5/K
theta <- theta + alpha # smooth
doc_topics <- t(theta / rowSums(theta)) # normalize


# create dataframes
df <- as.data.frame(doc_topics)
get_id <- function(x) sub("^([^.]*).*", "\\1", basename(x))
fileids <- sapply(rownames(df), get_id)
map <- as.data.frame(readRDS(qid_path))
map <- map[map$File_ID %in% fileids,]
df <- df[match(fileids, map$QID),]
df$QID <- map$QID

fileids <- sapply(rownames(df), get_id)
qid <- as.data.frame(readRDS(qid_path))
match(qid$File_ID, df$File_ID)

# initialize a mongodb connection and remove existing docs in collection
m <- mongo("docs.topics", url = mongo_url)
m$remove('{}')


# convert dataframe rows to json strings
row_to_json <- function(row) {
    to_list <- function(x) list(topicId = unbox(x), 
                                probability = unbox(row[[x]]))
    topics <- lapply(1:(length(row)-1), to_list)
    data <- list(
        `_id` = unbox(as.numeric(row[[length(row)]])),
        topics = topics
    )
    toJSON(data, digits = NA)
}
docs <- apply(df, 1, row_to_json)


# insert all documents
m$insert(docs)