library(RMySQL)

path <- "/dsl/eebo/wordembeddings/2020.03.02-decades/results/alignment.txt"

df <- read.csv(path, encoding="UTF-8", sep = " ", header=FALSE,
               stringsAsFactors = FALSE)

# reverse the order of columns
df <- df[,order(c(1, ncol(df):2))]

colnames <- c("word", seq(1470, 1700, 10))
colnames(df) <- colnames

con <- dbConnect(MySQL(),
                 user = "avkoehl",
                 password = "E]h+65S<5t395=!k",
                 dbname = "EEBO_Models",
                 host = "127.0.0.1")

dbWriteTable(con, "timeseries", df, overwrite = TRUE, append = FALSE,
             row.names = FALSE)
