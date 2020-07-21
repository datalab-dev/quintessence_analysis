library(mongolite)
library(jsonlite)

MONGO_URL <- "mongodb://localhost:27017"
META_PATH <- "../tmp/eebo_phase1_2_meta.csv"
QID_PATH <- "../tmp/qid.rds"


is.digit <- function(x) {
    suppressWarnings(!is.na(as.numeric(x)))
}


# read in metadata csv
meta <- read.csv(META_PATH, row.names = NULL, stringsAsFactors = FALSE)

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
qid <- as.data.frame(readRDS(QID_PATH))
meta <- meta[match(qid$File_ID, meta$File_ID),]
meta$QID <- qid$QID

# metadata
m <- mongo("docs.metadata", url = MONGO_URL)
df <- meta[c("QID", "Title", "Author", "Location", "Publisher", "Date", "Decade", "Word_Count", "Keywords", "Language")]
colnames(df) <- c("_id", "title", "author", "location", "publisher", "date", "decade", "wordCount", "keywords", "language")
m$remove('{}')
m$insert(df)

# ref ids
m <- mongo("docs.refids", url = MONGO_URL)
df <- meta[c("QID", "File_ID", "STC_ID", "ESTC_ID", "EEBO_Citation", "Proquest_ID", "VID")]
colnames(df) <- c("_id", "fileId", "stcId", "estcId", "eeboCitation", "proquestId", "vid")
m$remove('{}')
m$insert(df)