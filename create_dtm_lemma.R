library(RMySQL)
library(tm)
library(topicmodels) #for dtm2ldaformat
library(lda)
library(parallel)

odir = "/dsl/eebo/topicmodels/2020.02.19/"
english_only_fpath = "/dsl/eebo/2020.02.03-phase1-phase2/metadata/eebo_english.rds"
ncores = 20 # for cleaning
english_only = TRUE
sparsity = 0.995

log = function(msg) {
    print(paste(Sys.time(), msg))
}

myToLower <- function(x) {
    .Internal(tolower(x))
}

# load all the documents from the database
log("Getting Documents from the Database")
eebo_con = dbConnect(MySQL(), user="avkoehl", password="E]h+65S<5t395=!k",
                     dbname="quintessence_corpus", host="127.0.0.1")
query = "SELECT File_ID,Lemma FROM Lemma"
results = dbSendQuery(eebo_con, 'set character set "utf8"')
results = dbSendQuery(eebo_con, query)
docs = fetch(results, n=-1)

# subset the dataframe if english only topic model
if (english_only == TRUE) {
    subset = readRDS(english_only_fpath)
    ids = as.character(subset$File_ID)
    docs = docs[docs$File_ID %in% ids,]
}

# create corpus object
log("creating Corpus object")
docs = lapply(docs, function(x) gsub( "\t", " ", x))
docs_lemma = docs$Lemma
names(docs_lemma) = docs$File_ID
corpus = Corpus(VectorSource(docs_lemma))

# cleaning
cl = makeCluster(ncores)
tm_parLapply_engine(cl)
log("removing punctuation")
corpus$content = tm_parLapply(corpus, removePunctuation)
log("removing numbers")
corpus$content = tm_parLapply(corpus, removeNumbers)
log("removing stopwords")
stopwords <- union(readLines("../data/stopwords.txt"), stopwords(kind = "en"))
log("replacing pipes")
replace_pipes <- content_transformer(function(x)
    gsub(x, pattern = "|", replacement = " ", fixed = TRUE)
)
clusterExport(cl, "replace_pipes")
corpus = tm_map(corpus, replace_pipes)
tm_parLapply_engine(NULL)
stopCluster(cl)

# create dtm and filter out infrequent words and stopwords
log("create dtm and filter sparse terms")
corpus$content = unlist(corpus$content)
full_dtm = DocumentTermMatrix(corpus, control = list(tolower = myToLower))
log(sprintf("vocab size: %d", ncol(full_dtm)))
dtm = removeSparseTerms(full_dtm, sparsity)
log(sprintf("vocab size w/ %f sparsity: %d", sparsity, ncol(dtm)))
dtm <- dtm[, !colnames(dtm) %in% stopwords]

saveRDS(corpus, file=paste(odir, "corpus.rds", sep = ""))
saveRDS(dtm, file=paste(odir, "dtm.rds", sep = ""))
saveRDS(full_dtm, file=paste(odir, "full_dtm.rds", sep = ""))
