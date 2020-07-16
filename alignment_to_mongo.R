library(mongolite)
library(jsonlite)


PATH <- "../tmp/alignment.txt"
MONGO_URL <- "mongodb://localhost:27017"


# read the alignments text file
df <- read.csv(PATH, encoding="UTF-8", sep = " ", header=FALSE,
               stringsAsFactors = FALSE)

# reverse the order of the year columns for graphing
df <- df[,order(c(1, ncol(df):2))]

# initialize a mongodb connection and remove existing docs in collection
m <- mongo("timeseries", url = MONGO_URL)
m$remove('{}')

# convert dataframe rows to json strings
row_to_json <- function(row) {
    data <- list(
        `_id` = unbox(row[1]),
        timeseries = as.numeric(row[2:length(row)])
    )
    toJSON(data, digits = NA)
}
docs <- apply(df, 1, row_to_json)

# insert all documents
m$insert(docs)