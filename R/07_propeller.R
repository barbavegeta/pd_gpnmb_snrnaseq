#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript R/07_propeller.R <cell_composition_input.csv> <output.csv>")
}

input <- args[[1]]
output <- args[[2]]

suppressPackageStartupMessages({
  library(speckle)
})

x <- read.csv(input, stringsAsFactors = FALSE)
required <- c("sample", "group", "cell_type")
if (!all(required %in% colnames(x))) {
  stop(paste("Input must contain:", paste(required, collapse = ", ")))
}

x <- x[complete.cases(x[, required]), ]
x$sample <- factor(x$sample)
x$group <- factor(x$group, levels = c("Control", "PD"))
x$cell_type <- factor(x$cell_type)

# Propeller performs sample-aware tests on transformed cell-type proportions.
# This is preferable to independent t-tests on raw proportions, which ignore
# the mean-variance relationship and biological replication structure.
res <- propeller(
  clusters = x$cell_type,
  sample = x$sample,
  group = x$group,
  robust = TRUE,
  transform = "logit"
)

write.csv(res, output, row.names = TRUE)
print(res)
