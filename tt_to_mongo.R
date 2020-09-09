library(mongolite)
library(jsonlite)


mongo_url <- "mongodb://localhost:27017"
tt_path <- "../tmp/topicterms-75.rds"
beta <- 0.01  # topic model param


# terms topics matrix
phi <- as.data.frame(readRDS(tt_path))
phi <- phi + beta # smooth
df <- t(phi / rowSums(phi))


# initialize a mongodb connection and remove existing docs in collection
m <- mongo("terms.topics", url = mongo_url)
m$remove('{}')

# convert dataframe rows to json strings
row_to_json <- function(i) {
    row <- df[i,]
    to_list <- function(j) list(topic = unbox(j), 
                                probability = unbox(df[i,j]))
    topics <- lapply(1:ncol(df), to_list)
    
    data <- list(
        `_id` = unbox(rownames(df)[i]),
        topics = topics
    )
    toJSON(data, digits = NA)
}
docs <- sapply(1:nrow(df), row_to_json)

# insert all documents
m$insert(docs)
