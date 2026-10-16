# Simulates trees from the ADB density for the likelihood tests (profile likelihood curves)
#
# Usage (from the repository root): Rscript scripts/simulate_likelihood_test_trees.R [n_trees] [config ...] [overwrite]
#   Configurations whose files exist are skipped, so that profile likelihood results stay matched to their trees; add
#   the word overwrite to simulate them again, e.g. Rscript scripts/simulate_likelihood_test_trees.R 100 adb_1type_shape1 overwrite
#
# Per configuration, trees are simulated with sim_adb_origin_samp from a fixed origin of type 0, with the hidden
# divisions kept (collapse = FALSE). There is no window on the number of tips: the trees are draws from the density
# conditioned on at least two sampled tips (min_taxa = 2; scTreeSim loses the hidden divisions of trees with a single
# tip), which differs from the conditioning on survival in ADBTreeDistribution only by the small probability of a
# single tip. The origin time is chosen per shape so that the trees have tens of tips (median about 20). The two types
# differ in lifetime and death probability, so that P0 depends on the type. Taxon labels are a random permutation of
# 1..n.
#
# Output per configuration:
#   test_data/<config>.newick            annotated trees (hidden divisions as single-child nodes), one per line
#   test_data/<config>_collapsed.newick  the same trees with the single-child nodes collapsed (ape::collapse.singles,
#                                        as in treeio::drop.tip with collapse.singles = TRUE), tip types as [&type=..]
#   test_data/<config>.txt               parameters, seed, tips
#
# Note: treeio::write.beast.newick does not annotate the root node, so the origin type must be 0.

suppressPackageStartupMessages({
  library(scTreeSim)
  library(ape)
  library(treeio)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
overwrite <- "overwrite" %in% args # without it, existing output is kept: the results belong to these trees
args <- setdiff(args, "overwrite")
n_trees <- if (length(args) > 0) as.integer(args[1]) else 100L
out_dir <- "test_data"

# one type, or two types: type 0 divides into {0, 0} (0.7) or {0, 1} (0.3), type 1 only into {1, 1}
one_type <- list(ntypes = 1, mean_lifetime = 1, death_prob = 0.1,
                 sym_trans_prob = matrix(1), asym_trans_prob = matrix(0))
two_types <- list(ntypes = 2, mean_lifetime = c(1, 0.6), death_prob = c(0.1, 0.3),
                  sym_trans_prob = matrix(c(0.7, 0, 0, 1), nrow = 2, byrow = TRUE),
                  asym_trans_prob = matrix(c(0, 0.3, 0, 0), nrow = 2, byrow = TRUE))

# the origin time per shape gives a median of about 20 tips
configs <- list()
seed <- 30000
for (shape in c(1, 10, 100)) {
  origin_time <- if (shape == 1) 7 else 9
  for (model in list(list("1type", one_type), list("2type", two_types))) {
    seed <- seed + 1
    configs[[paste0("adb_", model[[1]], "_shape", shape)]] <-
      c(model[[2]], list(shape = shape, sampling_prob = 0.1, origin_time = origin_time, seed = seed))
  }
}
if (length(args) > 1) configs <- configs[args[-1]]

n_tips <- function(tree) length(tree@phylo$tip.label)

# the tree without single-child nodes, as a Newick string with the tip types
collapsed_newick <- function(tree) {
  tips <- tree@phylo$tip.label
  types <- tree@data$type[match(seq_along(tips), tree@data$node)] # the tips are the nodes 1..n
  phy <- ape::collapse.singles(tree@phylo)
  phy$tip.label <- paste0(tips, "[&type=", types, "]")
  phy$root.edge <- tree@phylo$origin - max(ape::node.depth.edgelength(phy))
  gsub("'", "", ape::write.tree(phy), fixed = TRUE)
}

simulate_config <- function(name, cfg) {
  set.seed(cfg$seed)
  trees <- vector("list", n_trees)
  n_draws <- 0

  while (n_draws < n_trees) {
    tree <- suppressMessages(sim_adb_origin_samp(
      origin_time = cfg$origin_time, shape = rep(cfg$shape, cfg$ntypes), mean_lifetime = cfg$mean_lifetime,
      death_prob = cfg$death_prob, sampling_prob = cfg$sampling_prob, asym_trans_prob = cfg$asym_trans_prob,
      sym_trans_prob = cfg$sym_trans_prob, min_taxa = 2, collapse = FALSE))
    if (is.null(tree)) next

    n <- n_tips(tree)
    tree@phylo$tip.label <- as.character(sample(n))
    tree@data$status <- NULL

    n_draws <- n_draws + 1
    trees[[n_draws]] <- tree
  }

  dir.create(out_dir, showWarnings = FALSE)
  newick_file <- file.path(out_dir, paste0(name, ".newick"))
  lines <- vapply(trees, function(tree) {
    f <- tempfile()
    treeio::write.beast.newick(tree, f)
    paste(readLines(f), collapse = "")
  }, character(1))
  writeLines(lines, newick_file)
  writeLines(vapply(trees, collapsed_newick, character(1)), file.path(out_dir, paste0(name, "_collapsed.newick")))

  sizes <- vapply(trees, n_tips, integer(1))
  info <- c(
    paste("config:", name),
    paste("seed:", cfg$seed),
    paste("n_trees:", n_trees),
    paste("tips: min", min(sizes), ", median", median(sizes), ", mean", round(mean(sizes), 2), ", max", max(sizes)),
    paste("origin_time:", cfg$origin_time),
    paste("shape:", cfg$shape),
    paste("mean_lifetime (per type):", paste(cfg$mean_lifetime, collapse = ",")),
    paste("death_prob (per type):", paste(cfg$death_prob, collapse = ",")),
    paste("sampling_prob:", cfg$sampling_prob),
    paste("sym_trans_prob (row-wise):", paste(t(cfg$sym_trans_prob), collapse = ",")),
    paste("asym_trans_prob (row-wise):", paste(t(cfg$asym_trans_prob), collapse = ","))
  )
  writeLines(info, file.path(out_dir, paste0(name, ".txt")))
  message(name, ": ", info[4])
}

for (name in names(configs)) {
  file <- file.path(out_dir, paste0(name, ".newick"))
  if (file.exists(file) && !overwrite) message(file, " exists, skipped (add overwrite to repeat it)")
  else simulate_config(name, configs[[name]])
}
