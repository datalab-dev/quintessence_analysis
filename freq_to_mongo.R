library(mongolite)
library(jsonlite)


DIR <- "../tmp/2020.03.02"
MONGO_URL <- "mongodb://localhost:27017"

fields <- c("full", "decades", "locations", "authors")


# Log to console
log <- function(msg) {
    print(paste(Sys.time(), msg, sep = " "))
}


# Get the name of the author/location/decade from a filename
get_basename <- function(fname, isdecade = FALSE) {
    tmp <- strsplit(fname, split = "/|\\.")[[1]]
    tmp <- tmp[length(tmp) - 1]
    if (isdecade == TRUE) tmp <- paste(tmp, "0", sep = "")
    tmp
}

# Given a row of a nearest neighbors dataframe upsert it into the database
upsert_row <- function(row, field, name) {
    # construct key
    key <- list(`_id` = unbox(as.character(row[1])))
    key_json <- toJSON(key)
    
    # construct new object
    if (field == "full")
        name <- NULL
    
    upsert_loc <- paste(c("frequencies", field, name), collapse = ".")
    data <- list(`$set` = list())
    data$`$set`[[upsert_loc]] <- list(
        freq = unbox(as.numeric(unlist(row[2]))),
        relFreq = unbox(as.numeric(unlist(row[3])))
    )
    data_json <- toJSON(data, digits = NA)
    
    # upsert object
    m$update(key_json, data_json, upsert = TRUE)
}

# Given full/decade/author/location insert nearest neighbors into database
nn_to_mongo <- function(field) {
    # get nn filenames
    path <- if (field == "full") DIR else sprintf("%s-%s/results", DIR, field)
    nn_files <- list.files(path, pattern = "*.freq", full.names = TRUE)
    
    # process each row in each file
    process_file <- function(fname) {
        log(fname)
        df <- read.csv(fname, encoding = "UTF-8", sep = " ", header = FALSE,
                       stringsAsFactors = FALSE)
        isdecade <- if (field == "decades") TRUE else FALSE
        name <- get_basename(fname, isdecade)
        apply(df, 1, upsert_row, field, name)
    }
    sapply(nn_files, process_file)
}


# initialize a mongodb connection
m <- mongo("terms.frequencies", url = MONGO_URL)

# add to db
sapply(fields, nn_to_mongo)