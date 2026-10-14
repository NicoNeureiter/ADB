# Simulates annotated trees from the ADB density conditioned on the number of tips and the tip types,
# for the detailed-balance tests (see dev/detailed-balance-tests.md).
#
# Usage (from the repository root): Rscript scripts/simulate_test_trees.R [n_trees]
#
# Per configuration, trees are simulated with a fixed origin, hidden divisions are kept (collapse = FALSE),
# and only trees with exactly n tips and the required tip-type counts are kept (exact conditioning by rejection).
# Taxon labels are then assigned at random within each type. Output: test_data/<config>.newick (one tree per
# line) and test_data/<config>.txt (parameters, seed, acceptance rate).
#
# Note: treeio::write.beast.newick does not annotate the root node, so the origin type must be 0.

suppressPackageStartupMessages({
  library(scTreeSim)
  library(ape)
  library(treeio)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
n_trees <- if (length(args) > 0) as.integer(args[1]) else 2000L
out_dir <- "test_data"

configs <- list(
  adb_1type_n4 = list(
    seed = 20241,
    origin_time = 3,
    shape = 3, mean_lifetime = 1, death_prob = 0.1, sampling_prob = 0.5,
    sym_trans_prob = matrix(1), asym_trans_prob = matrix(0),
    tip_types = c(4)                # number of tips per type (type 0, 1, ...)
  ),
  adb_2type_n4 = list(
    seed = 20242,
    origin_time = 3,
    shape = c(3, 3), mean_lifetime = c(1, 1), death_prob = c(0.1, 0.1), sampling_prob = 0.5,
    # type 0 divides into {0, 0} (0.7) or {0, 1} (0.3), type 1 only into {1, 1}
    sym_trans_prob = matrix(c(0.7, 0, 0, 1), nrow = 2, byrow = TRUE),
    asym_trans_prob = matrix(c(0, 0.3, 0, 0), nrow = 2, byrow = TRUE),
    tip_types = c(2, 2)
  )
)

n_tips <- function(tree) length(tree@phylo$tip.label)

simulate_config <- function(name, cfg) {
  set.seed(cfg$seed)
  n <- sum(cfg$tip_types)
  ntypes <- length(cfg$tip_types)
  trees <- vector("list", n_trees)
  n_draws <- 0
  n_kept <- 0

  while (n_kept < n_trees) {
    n_draws <- n_draws + 1
    # same as sim_adb_origin_samp, split in two so that draws with < n alive particles are rejected early
    full <- suppressMessages(sim_adb_origin_complete_fast(
      origin_time = cfg$origin_time, shape = cfg$shape, mean_lifetime = cfg$mean_lifetime,
      death_prob = cfg$death_prob, asym_trans_prob = cfg$asym_trans_prob,
      sym_trans_prob = cfg$sym_trans_prob, min_taxa = n))
    if (is.null(full)) next
    tree <- suppressMessages(scTreeSim:::prune_tree(full, sampling_prob = cfg$sampling_prob, min_taxa = n,
                                                    collapse = FALSE))
    if (is.null(tree) || n_tips(tree) != n) next

    tips <- tree@phylo$tip.label
    types <- tree@data$type[match(seq_along(tips), tree@data$node)]
    if (!identical(as.integer(tabulate(types + 1, nbins = ntypes)), as.integer(cfg$tip_types))) next

    # random labels within each type: type t gets the labels in its own block of consecutive numbers
    labels <- character(n)
    first <- cumsum(c(0, cfg$tip_types))
    for (t in seq_len(ntypes) - 1) {
      idx <- which(types == t)
      labels[idx] <- sample(first[t + 1] + seq_along(idx))
    }
    tree@phylo$tip.label <- labels
    tree@data$status <- NULL

    n_kept <- n_kept + 1
    trees[[n_kept]] <- tree
  }

  dir.create(out_dir, showWarnings = FALSE)
  newick_file <- file.path(out_dir, paste0(name, ".newick"))
  lines <- vapply(trees, function(tree) {
    f <- tempfile()
    treeio::write.beast.newick(tree, f)
    paste(readLines(f), collapse = "")
  }, character(1))
  writeLines(lines, newick_file)

  info <- c(
    paste("config:", name),
    paste("seed:", cfg$seed),
    paste("n_trees:", n_trees),
    paste("acceptance:", signif(n_kept / n_draws, 3), "(", n_kept, "of", n_draws, "draws )"),
    paste("origin_time:", cfg$origin_time),
    paste("shape:", paste(cfg$shape, collapse = ",")),
    paste("mean_lifetime:", paste(cfg$mean_lifetime, collapse = ",")),
    paste("death_prob:", paste(cfg$death_prob, collapse = ",")),
    paste("sampling_prob:", cfg$sampling_prob),
    paste("sym_trans_prob (row-wise):", paste(t(cfg$sym_trans_prob), collapse = ",")),
    paste("asym_trans_prob (row-wise):", paste(t(cfg$asym_trans_prob), collapse = ",")),
    paste("tip_types (tips per type):", paste(cfg$tip_types, collapse = ","))
  )
  writeLines(info, file.path(out_dir, paste0(name, ".txt")))
  message(name, ": ", paste(info[4], collapse = " "))
}

for (name in names(configs)) simulate_config(name, configs[[name]])
