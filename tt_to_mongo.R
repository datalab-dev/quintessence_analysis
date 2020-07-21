library(mongolite)
library(jsonlite)


MONGO_URL <- "mongodb://localhost:27017"
TT_PATH <- "../tmp/topicterms-75.rds"


# topic terms matrix
df <- as.data.frame(readRDS(TT_PATH))
df$topic_num <- seq.int(nrow(df))

# initialize a mongodb connection and remove existing docs in collection
m <- mongo("topics.terms", url = MONGO_URL)
m$remove('{}')

# convert dataframe rows to json strings
row_to_json <- function(row) {
    freqs <- row[2:length(row) - 1]
    to_list <- function(x) list(term = unbox(x), frequency = unbox(freqs[[x]]))
    topics <- lapply(names(freqs), to_list)
    
    data <- list(
        `_id` = unbox(unlist(row["topic_num"])),
        topics = topics
    )
    toJSON(data, digits = NA)
}
docs <- apply(df, 1, row_to_json)

# insert all documents
m$insert(docs)
