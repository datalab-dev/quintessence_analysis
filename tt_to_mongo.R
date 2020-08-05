library(mongolite)
library(jsonlite)


mongo_url <- "mongodb://localhost:27017"
tt_path <- "../tmp/topicterms-75.rds"
beta <- 0.01  # topic model param


# topic terms matrix
phi <- as.data.frame(readRDS(tt_path))
phi <- phi + beta # smooth
df <- phi / rowSums(phi) # normalize
df$topic_num <- seq.int(nrow(df))

# initialize a mongodb connection and remove existing docs in collection
m <- mongo("topics.terms", url = mongo_url)
m$remove('{}')

# convert dataframe rows to json strings
row_to_json <- function(row) {
    freqs <- row[2:length(row) - 1]
    to_list <- function(x) list(term = unbox(x), 
                                probability = unbox(freqs[[x]]))
    terms <- lapply(names(freqs), to_list)
    
    data <- list(
        `_id` = unbox(unlist(row["topic_num"])),
        terms = terms
    )
    toJSON(data, digits = NA)
}
docs <- apply(df, 1, row_to_json)

# insert all documents
m$insert(docs)
