library(RMySQL)


# Configuration
odir <- "/dsl/eebo/wordembeddings/2020.03.02"
# fields <- c("full", "decade", "location", "author")
fields <- c("location", "author")


# Log to console
log <- function(msg) {
    print(paste(Sys.time(), msg, sep = " "))
}


# Get the basename from a filename
get_basename <- function(fname, isdecade = FALSE) {
    tmp <- strsplit(fname, split = "/|\\.")[[1]]
    tmp <- tmp[length(tmp) - 1]
    if (isdecade == TRUE) tmp <- paste(tmp, "0", sep = "")
    tmp
}


# Get the table name from a filename based on the type of model
get_tname <- function(fname, ttype, model) {
    if (model == "decade") {
        tname <- paste(get_basename(fname, TRUE), "_", ttype, sep = "")
    } else if (model == "full") {
        tname <- paste("full_", ttype, sep = "")
    } else {
        tname <- paste(get_basename(fname), "_", ttype, sep = "")
        tname <- gsub("-", "_", tname)
    }

    tname
}


# Create a table for a nearest neighbors file
nn_to_sql <- function(fname, con, model) {
    data <- read.table(fname, header = FALSE, stringsAsFactors = FALSE)
    n <- (ncol(data) - 1) / 2
    colnames <- paste(rep(c("neighbor", "score"), n),
                      rep(1:n, each = 2),
                      sep = "")
    colnames <- c("word", colnames)
    colnames(data) <- colnames
    tname <- get_tname(fname, "neighbors", model)
    log(tname)
    dbWriteTable(con, tname, data, overwrite = TRUE, append = FALSE,
                 row.names = FALSE)
}


# Create a table for a frequencies file
freq_to_sql <- function(fname, con, model) {
    data <- read.table(fname, header = FALSE, stringsAsFactors = FALSE)
    colnames <- c("word", "freq", "rel_freq")
    colnames(data) <- colnames
    tname <- get_tname(fname, "freq", model)
    log(tname)
    dbWriteTable(con, tname, data, overwrite = TRUE, append = FALSE,
                 row.names = FALSE)
}


con <- dbConnect(MySQL(),
                 user = "avkoehl",
                 password = "E]h+65S<5t395=!k",
                 dbname = "EEBO_Models",
                 host = "127.0.0.1")


for (field in fields) {
    log(field)

    dir <- odir
    if (field != "full")
        dir <- paste0(odir, "-", field, "s/results")

    nn_files <- list.files(dir, pattern = "*.neighbors", full.names = TRUE)
    freq_files <- list.files(dir, pattern = "*.freq", full.names = TRUE)
    sapply(nn_files, nn_to_sql, con, field)
    sapply(freq_files, freq_to_sql, con, field)
}
