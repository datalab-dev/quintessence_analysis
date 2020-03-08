# Test a vector of sparsity values for resulting vocabulary size

library(tm)

sparsity_vec <- seq(0.995, 1, 0.0005)[1:10]
full_dtm <- readRDS("/dsl/eebo/topicmodels/2020.02.19/full_dtm.rds")

vocab_size <- function(sparsity) {
    tmp <- removeSparseTerms(full_dtm, sparsity)
    ncol(tmp)
}

sizes <- sapply(sparsity_vec, vocab_size)

df <- data.frame(sparsity = sparsity_vec, vocab_size = sizes)
print(df)
