library(RMySQL)


# config
meta_path <- "../data/eebo_meta.csv"
con <- dbConnect(MySQL(),
                 user = "avkoehl",
                 password = "E]h+65S<5t395=!k",
                 dbname = "quintessence",
                 host = "127.0.0.1")


is.digit <- function(x) {
    suppressWarnings(!is.na(as.numeric(x)))
}


meta <- read.csv(meta_path, row.names = NULL, stringsAsFactors = FALSE)
dates <- meta[["Date"]]

decades <- list()
i <- 1

for (date in dates) {
    if (!is.digit(date) || as.numeric(date) < 1470 || as.numeric(date) > 1700) {
        decades[[i]] <- 0
    } else {
        decades[[i]] <- substr(date, start = 1, stop = 3)
    }
    i <- i + 1
}

meta <- cbind(meta, as.data.frame(unlist(decades)))
names(meta)[ncol(meta)] <- "Decade"

dbWriteTable(con, "EEBO_Metadata", meta, overwrite = TRUE, append = FALSE,
             row.names = FALSE)
