library(RMySQL)

# config
tt_path <- "/dsl/eebo/topicmodels/2019.09.24/topic_terms.rds"
dt_path <- "/dsl/eebo/topicmodels/2019.09.24/doc_topics.rds"

con <- dbConnect(MySQL(),
                 user = "avkoehl",
                 password = "E]h+65S<5t395=!k",
                 dbname = "EEBO_Models",
                 host = "127.0.0.1")

# topic terms matrix
topic_terms <- as.data.frame(readRDS(tt_path))
dbWriteTable(con, "topic_terms", topic_terms, overwrite = TRUE, append = FALSE,
             row.names = FALSE)
print("wrote topic terms")

# doc topics matrix
doc_topics <- as.data.frame(readRDS(dt_path))
dbWriteTable(con, "doc_topics", doc_topics, overwrite = TRUE, append = FALSE,
             row.names = FALSE)
print("wrote doc topics")
