# Simulates trees for the runtime and error profiling of the likelihood (test/adb/distribution/RuntimeAndErrorProfiling.java)
#
# Usage (from the repository root): Rscript scripts/simulate_profiling_trees.R [experiment ...] [overwrite]
#   Experiments whose file exists are skipped, so that profiling results stay matched to their trees; add the word
#   overwrite to simulate them again, e.g. Rscript scripts/simulate_profiling_trees.R size overwrite
#
# Output: one tab-separated file per experiment in profiling/trees/ (git-ignored), one row per tree, with the columns
#   config, tree, ntypes, graph, shape, lifetime, death, rho, origin_time, tips, newick_annotated, newick_collapsed
# where newick_annotated has the hidden divisions as single-child nodes and newick_collapsed has them removed, with
# tip types as [&type=..]. The profiling reads the model parameters from the same row (all types share lifetime and
# death probability, since the cost does not depend on that). Experiments:
#   size         tips 10, 30, 100, 300, 1000, 3000 at the base model (fixed number of taxa; the origin is whatever the
#                simulation ends with, so it grows with the size)
#   ntypes       1, 2, 3, 5, 10 types, chain and dense transition graph, 100 tips
#   shape        shape 1, 2, 5, 20, 100, one type, 100 tips
#   rho_origin   sampling proportion 1, 0.1, 0.01, 0.001 at a fixed origin (13), so the number of tips follows rho
#   rho_taxa     sampling proportion 1, 0.1, 0.01, 0.001 with 100 tips, so the origin follows rho
#
# Base model: one type, shape 10, mean lifetime 1, death probability 0.1, sampling probability 0.1.
# Transition graphs for n types (type 0 is the origin type): chain, type i gives rise to type i+1 with the asymmetric
# probability 0.2 (a division into i and i+1), else divides into i and i (the last type only into itself); dense, every
# other type with the asymmetric probability 0.2 / (n - 1). The root of the tree is not annotated by
# treeio::write.beast.newick, so the origin type is 0.
#
# Why not the plain `Rscript` default of sim_adb_origin_samp with a window on the tips: for the fixed-origin
# experiments the number of tips is part of what is measured; trees with more than 5000 tips are redrawn.

suppressPackageStartupMessages({
  library(scTreeSim)
  library(ape)
  library(treeio)
})

args <- commandArgs(trailingOnly = TRUE)
overwrite <- "overwrite" %in% args # without it, existing output is kept: the results belong to these trees
args <- setdiff(args, "overwrite")
out_dir <- "profiling/trees"
max_tips <- 5000
origin_for_rho <- 13

base <- list(ntypes = 1, graph = "none", shape = 10, lifetime = 1, death = 0.1, rho = 0.1)

transitions <- function(ntypes, graph) {
  sym <- diag(ntypes)
  asym <- matrix(0, ntypes, ntypes)
  if (ntypes > 1) {
    for (i in seq_len(ntypes)) {
      if (graph == "chain") {
        if (i < ntypes) { asym[i, i + 1] <- 0.2; sym[i, i] <- 0.8 }
      } else {
        asym[i, -i] <- 0.2 / (ntypes - 1)
        sym[i, i] <- 0.8
      }
    }
  }
  list(sym = sym, asym = asym)
}

# each entry: a model (base with changes), the number of taxa or the origin, the number of trees, and a seed
experiments <- list()
seed <- 40000
add <- function(experiment, config, changes, n_trees, ntaxa = NA, origin = NA) {
  seed <<- seed + 1
  experiments[[experiment]][[config]] <<- list(
    model = modifyList(base, changes), n_trees = n_trees, ntaxa = ntaxa, origin = origin, seed = seed)
}

for (n in c(10, 30, 100, 300, 1000, 3000)) add("size", paste0("tips", n), list(), 5, ntaxa = n)
add("ntypes", "types1", list(), 5, ntaxa = 100)
for (graph in c("chain", "dense")) {
  for (k in c(2, 3, 5, 10)) add("ntypes", paste0("types", k, "_", graph), list(ntypes = k, graph = graph), 5, ntaxa = 100)
}
for (shape in c(1, 2, 5, 20, 100)) add("shape", paste0("shape", shape), list(shape = shape), 5, ntaxa = 100)
for (rho in c(1, 0.1, 0.01, 0.001)) {
  add("rho_origin", paste0("rho", rho), list(rho = rho), 5, origin = origin_for_rho)
  add("rho_taxa", paste0("rho", rho), list(rho = rho), 5, ntaxa = 100)
}

n_tips <- function(tree) length(tree@phylo$tip.label)

simulate_tree <- function(cfg) {
  m <- cfg$model
  tm <- transitions(m$ntypes, m$graph)
  shape <- rep(m$shape, m$ntypes)
  repeat {
    tree <- suppressMessages(
      if (is.na(cfg$ntaxa)) {
        sim_adb_origin_samp(origin_time = cfg$origin, shape = shape, mean_lifetime = rep(m$lifetime, m$ntypes),
                            death_prob = rep(m$death, m$ntypes), sampling_prob = m$rho, asym_trans_prob = tm$asym,
                            sym_trans_prob = tm$sym, min_taxa = 2, collapse = FALSE)
      } else {
        sim_adb_ntaxa_samp(ntaxa = cfg$ntaxa, shape = shape, mean_lifetime = rep(m$lifetime, m$ntypes),
                           death_prob = rep(m$death, m$ntypes), sampling_prob = m$rho, asym_trans_prob = tm$asym,
                           sym_trans_prob = tm$sym, collapse = FALSE)
      })
    if (!is.null(tree) && n_tips(tree) >= 2 && n_tips(tree) <= max_tips) return(tree)
  }
}

# the tree without single-child nodes, as a Newick string with the tip types
collapsed_newick <- function(tree) {
  tips <- tree@phylo$tip.label
  types <- tree@data$type[match(seq_along(tips), tree@data$node)] # the tips are the nodes 1..n
  phy <- ape::collapse.singles(tree@phylo)
  phy$tip.label <- paste0(tips, "[&type=", types, "]")
  phy$root.edge <- tree@phylo$origin - max(ape::node.depth.edgelength(phy))
  gsub("'", "", ape::write.tree(phy), fixed = TRUE)
}

annotated_newick <- function(tree) {
  f <- tempfile()
  on.exit(unlink(f))
  treeio::write.beast.newick(tree, f)
  paste(readLines(f), collapse = "")
}

simulate_experiment <- function(experiment) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  rows <- list()
  for (config in names(experiments[[experiment]])) {
    cfg <- experiments[[experiment]][[config]]
    set.seed(cfg$seed)
    for (k in seq_len(cfg$n_trees)) {
      tree <- simulate_tree(cfg)
      n <- n_tips(tree)
      tree@phylo$tip.label <- as.character(sample(n))
      tree@data$status <- NULL
      m <- cfg$model
      rows[[length(rows) + 1]] <- data.frame(
        config = config, tree = k, ntypes = m$ntypes, graph = m$graph, shape = m$shape, lifetime = m$lifetime,
        death = m$death, rho = m$rho, origin_time = tree@phylo$origin, tips = n,
        newick_annotated = annotated_newick(tree), newick_collapsed = collapsed_newick(tree))
      message(experiment, " ", config, " tree ", k, ": ", n, " tips, origin ", round(tree@phylo$origin, 2))
    }
  }
  write.table(do.call(rbind, rows), file.path(out_dir, paste0("trees_", experiment, ".tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

selected <- if (length(args) > 0) args else names(experiments)
for (experiment in selected) {
  file <- file.path(out_dir, paste0("trees_", experiment, ".tsv"))
  if (file.exists(file) && !overwrite) message(file, " exists, skipped (add overwrite to repeat it)")
  else simulate_experiment(experiment)
}
