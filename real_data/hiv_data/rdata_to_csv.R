# All data processing follows https://github.com/lihualei71/dbh
library(tidyverse)
# Load data
rfiles <- c("HIV_data.RData", "HIV_discoveries.RData", "HIV_res.RData")
for (file in rfiles) {
  load(file)
}

# Save raw data as CSV
drug_types <- c("PI", "NRTI", "NNRTI")
for (j in 1:length(data)) {
  dataj <- data[[j]]
  resistances <- dataj[[1]]
  mutations <- dataj[[2]]
  mut_file <- paste(drug_types[j], "mutations.csv", sep="_")
  write.csv(mutations, mut_file)
  resist_file <- paste(drug_types[j], "resistances.csv", sep="_")
  write.csv(resistances, resist_file)
}

# Get positions of gene list and TSM gene list for each drug
signal_genes <- list()
for (drug_class in c("PI", "NRTI", "NNRTI")){
  base_url <- 'http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006'
  tsm_url <- paste(base_url, 'MUTATIONLISTS', 'NP_TSM', drug_class, sep = '/')
  tsm_df <- read.delim(tsm_url, header = FALSE, stringsAsFactors = FALSE)
  signal_genes[[drug_class]] <- rep(list(tsm_df[, 1]),
                                    length(data[[drug_class]]$Y))
}
library(jsonlite)
#signal_genes <- do.call(c, signal_genes)
sgjson = toJSON(signal_genes,pretty=TRUE,auto_unbox=TRUE)
write(sgjson, "signal_genes.json")
