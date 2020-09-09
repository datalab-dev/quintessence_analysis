library(mongolite)
library(jsonlite)
library(purrr)


mongo_url <- "mongodb://localhost:27017"
meta_path <- "../tmp/eebo_phase1_2_meta.csv"
qid_path <- "../tmp/qid.rds"


is.digit <- function(x) {
    suppressWarnings(!is.na(as.numeric(x)))
}


process_keywords = function(x) {
    x <- unlist(strsplit(x, "--"))
    x <- gsub("B.C", "BC", x, fixed=TRUE)
    x <- gsub("A.D", "AD", x, fixed=TRUE)
    x <- unlist(strsplit(x, ".", fixed=TRUE))
    trimws(x)
}


# read in metadata csv
meta <- read.csv(meta_path, row.names = NULL, stringsAsFactors = FALSE)

# process keywords
meta$Keywords <- lapply(meta$Keywords, process_keywords)

# add decade column
dates <- meta$Date
decades <- list()
i <- 1
for (date in dates) {
    if (!is.digit(date) || as.numeric(date) < 1470 || as.numeric(date) > 1700) {
        decades[[i]] <- 0
    } else {
        decades[[i]] <- paste0(substr(date, start = 1, stop = 3), "0")
    }
    i <- i + 1
}
meta$Date <- as.integer(meta$Date)
meta$Decade <- as.integer(unlist(decades))

# add id column
qid <- as.data.frame(readRDS(qid_path))
meta <- meta[match(qid$File_ID, meta$File_ID),]
meta$QID <- qid$QID

# metadata
m <- mongo("docs.metadata", url = mongo_url)
df <- meta[c("QID", "Title", "Author", "Location", "Publisher", "Date", 
             "Decade", "Word_Count", "Keywords", "Language", "File_ID", 
             "STC_ID", "ESTC_ID", "EEBO_Citation", "Proquest_ID", "VID")]
colnames(df) <- c("_id", "title", "author", "location", "publisher", "date", 
                  "decade", "wordCount", "keywords", "language", "fileId", 
                  "stcId", "estcId", "eeboCitation", "proquestId", "vid")


# json
meta_list <- transpose(df)
docs <- sapply(meta_list, toJSON, auto_unbox = TRUE)

# write to db
m <- mongo("docs.metadata", url = mongo_url)
m$remove('{}')
m$insert(docs, rownames = FALSE)