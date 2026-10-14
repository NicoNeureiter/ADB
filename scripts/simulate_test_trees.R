# Simulates annotated trees from the ADB density for the stationarity and flow tests
#
# Usage (from the repository root): Rscript scripts/simulate_test_trees.R [n_trees] [config ...]
#
# Per configuration, trees are simulated with a fixed origin, hidden divisions are kept (collapse = FALSE),
# and only trees with min_tips <= n <= max_tips tips are kept, in which every type occurs among the tips.
# The tree operators change neither the number of tips nor the tip types, so a window on these is valid
# conditioning: each (n, tip types) class is invariant by itself, hence so is their mixture.
# Taxon labels are a random permutation of 1..n. Output: test_data/<config>.newick (one tree per
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

# parameters shared by all configurations
common <- list(origin_time = 8, shape = 10, mean_lifetime = 1, death_prob = 0.1, sampling_prob = 0.1,
               min_tips = 4, max_tips = 16)

configs <- list(
  adb_1type_win = c(common, list(
    seed = 20251, ntypes = 1,
    sym_trans_prob = matrix(1), asym_trans_prob = matrix(0)
  )),
  adb_2type_win = c(common, list(
    seed = 20252, ntypes = 2,
    # type 0 divides into {0, 0} (0.7) or {0, 1} (0.3), type 1 only into {1, 1}
    sym_trans_prob = matrix(c(0.7, 0, 0, 1), nrow = 2, byrow = TRUE),
    asym_trans_prob = matrix(c(0, 0.3, 0, 0), nrow = 2, byrow = TRUE)
  )),
  adb_3type_win = c(common, list(
    seed = 20253, ntypes = 3,
    # type 0 divides into {0, 0} (0.6), {0, 1} (0.3) or {0, 2} (0.1), type 1 into {1, 1} (0.6) or {1, 2} (0.4),
    # type 2 only into {2, 2}
    sym_trans_prob = diag(c(0.6, 0.6, 1)),
    asym_trans_prob = matrix(c(0, 0.3, 0.1, 0, 0, 0.4, 0, 0, 0), nrow = 3, byrow = TRUE)
  ))
)
if (length(args) > 1) configs <- configs[args[-1]]

n_tips <- function(tree) length(tree@phylo$tip.label)

simulate_config <- function(name, cfg) {
  set.seed(cfg$seed)
  ntypes <- cfg$ntypes
  trees <- vector("list", n_trees)
  n_draws <- 0
  n_kept <- 0

  while (n_kept < n_trees) {
    n_draws <- n_draws + 1
    # same as sim_adb_origin_samp, split in two so that draws with too few alive particles are rejected early
    full <- suppressMessages(sim_adb_origin_complete_fast(
      origin_time = cfg$origin_time, shape = rep(cfg$shape, ntypes), mean_lifetime = rep(cfg$mean_lifetime, ntypes),
      death_prob = rep(cfg$death_prob, ntypes), asym_trans_prob = cfg$asym_trans_prob,
      sym_trans_prob = cfg$sym_trans_prob, min_taxa = cfg$min_tips))
    if (is.null(full)) next
    tree <- suppressMessages(scTreeSim:::prune_tree(full, sampling_prob = cfg$sampling_prob,
                                                    min_taxa = cfg$min_tips, collapse = FALSE))
    if (is.null(tree)) next
    n <- n_tips(tree)
    if (n < cfg$min_tips || n > cfg$max_tips) next

    tips <- tree@phylo$tip.label
    types <- tree@data$type[match(seq_along(tips), tree@data$node)]
    if (length(unique(types)) < ntypes) next

    tree@phylo$tip.label <- as.character(sample(n))
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

  sizes <- vapply(trees, n_tips, integer(1))
  info <- c(
    paste("config:", name),
    paste("seed:", cfg$seed),
    paste("n_trees:", n_trees),
    paste("acceptance:", signif(n_kept / n_draws, 3), "(", n_kept, "of", n_draws, "draws )"),
    paste("tips: window", cfg$min_tips, "to", cfg$max_tips, ", mean", round(mean(sizes), 2), ", every type among the tips"),
    paste("origin_time:", cfg$origin_time),
    paste("shape:", cfg$shape),
    paste("mean_lifetime:", cfg$mean_lifetime),
    paste("death_prob:", cfg$death_prob),
    paste("sampling_prob:", cfg$sampling_prob),
    paste("sym_trans_prob (row-wise):", paste(t(cfg$sym_trans_prob), collapse = ",")),
    paste("asym_trans_prob (row-wise):", paste(t(cfg$asym_trans_prob), collapse = ","))
  )
  writeLines(info, file.path(out_dir, paste0(name, ".txt")))
  message(name, ": ", paste(info[4], collapse = " "))
}

for (name in names(configs)) simulate_config(name, configs[[name]])
