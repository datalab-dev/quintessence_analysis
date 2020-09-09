library(mongolite)
library(jsonlite)
library(parallel)
library(data.table)


options(stringsAsFactors = FALSE)

fpath <- "../tmp/adorned/"
tpath <- "../tmp/adorned_truncated/"
qid_path <- "../tmp/qid.rds"
mongo_url <- "mongodb://localhost:27017"


###########################################################
#                  READ ADORNED FILES                     #
###########################################################

qids <- readRDS(qid_path)

print("making a cluster")
cluster <- makeCluster(40)

adorned <- list.files(path = fpath, pattern = "*.txt", recursive = TRUE, 
                      full.names = TRUE)
print(sprintf("num adorned files: %s", length(adorned)))

truncated <- list.files(path = tpath, pattern = "*.txt", recursive = TRUE, 
                        full.names = TRUE)
print(sprintf("num truncated files: %s", length(truncated)))


###########################################################
#      PARSE ADORNED FILES FOR POS, LEMMA, STANDARD       #
###########################################################

log <- function(msg) {
    print(paste(Sys.time(), msg, sep = " "))
}

get_content <- function(filename) {
    docname <- basename(filename)
    fileid <- sub("^([^.]*).*", "\\1", docname)
    
    eebo_adorn = tryCatch({
            read.table(filename, quote = "", sep = "\t", header = FALSE)
        },
        error = function(error) {
            return(c(fileid,"", "","",""))
        }
    )
    
    raw <- paste(eebo_adorn[[1]], collapse="\t")
    pos <- paste(eebo_adorn[[3]], collapse="\t")
    std <- tolower(paste(eebo_adorn[[4]], collapse="\t"))
    lemma <- paste(eebo_adorn[[5]], collapse="\t")
    
    raw <- tolower(raw)
    std <- tolower(std)
    c(fileid, raw, lemma, std, pos)
}


create_df <- function(filenames, qids) {
    results <- parLapply(cluster, filenames, get_content)
    df <- do.call(rbind.data.frame, results)
    colnames(df) <- c("File_ID", "raw", "lemma", "standardized", "partsOfSpeech")
    df <- merge(df, qids, by = "File_ID")
    colnames(df)[length(df)] <- "_id"
    df$File_ID <- NULL # don't need to store file id
    df
}


full_df <- create_df(adorned, qids)
m <- mongo("docs", url = mongo_url)
m$insert(full_df)
print("wrote the full df")

truncated_df <- create_df(truncated, qids)
m <- mongo("docs.truncated", url = mongo_url)
m$insert(truncated_df)
print("wrote the truncated df")