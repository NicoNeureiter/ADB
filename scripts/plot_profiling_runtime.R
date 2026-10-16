# Plots the run times written by RuntimeAndErrorProfiling (testRuntimeVs*)
#
# Usage (from the repository root): Rscript scripts/plot_profiling_runtime.R
#
# Reads profiling/results/runtime_and_error/runtime_{nsteps,size,ntypes}.csv and writes
# profiling/results/runtime_and_error/runtime_plots.pdf, one page per experiment. Seconds on a logarithmic axis with the
# same range and plain (not scientific) labels on all panels of a page, so that panels can be compared; the line is the
# median over the 5 trees, the ribbon the range; open symbols are extrapolated values (time budget exceeded).
#   page 1  nSteps (base model, 100 tips): P0, P1, the tree calculation and the full evaluation per mode
#   page 2  tree size (10 to 3000 tips): the tree calculation and the full evaluation per mode and nSteps
#   page 3  number of types, chain and dense graph (with the number of transition parameters): P0, P1, tree and full
# full = lifetimes + P0 + P1 + tree (annotated: no P1), the cost of an MCMC step after a model parameter changed.
# Modes: integrated (hidden divisions integrated out), approx (series approximation, one type), annotated.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

dir <- "profiling/results/runtime_and_error"
read_runtime <- function(experiment) {
  read.csv(file.path(dir, paste0("runtime_", experiment, ".csv"))) %>%
    mutate(extrapolated = as.logical(extrapolated))
}

mode_colours <- c(integrated = "#d95f02", approx = "#1b9e77", annotated = "#7570b3", all = "grey40")
plain <- function(x) format(x, scientific = FALSE, drop0trailing = TRUE, trim = TRUE)
seconds_axis <- scale_y_log10(breaks = 10^(-5:2), labels = plain, limits = c(3e-5, 200))
theme_runtime <- list(theme_bw(base_size = 11),
                      theme(legend.position = "bottom", plot.subtitle = element_text(size = 8),
                            panel.grid.minor = element_blank()),
                      scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 21), guide = "none"))

# median and range over the trees of the same configuration
summarise_trees <- function(d, ...) {
  d %>%
    group_by(...) %>%
    summarise(median = median(seconds), low = min(seconds), high = max(seconds),
              extrapolated = any(extrapolated), .groups = "drop")
}

pdf(file.path(dir, "runtime_plots.pdf"), width = 11, height = 7.5)

# page 1: nSteps
d <- read_runtime("nsteps") %>%
  filter(component %in% c("P0", "P1", "tree", "full"), !(component == "P1" & mode == "approx")) %>%
  mutate(series = ifelse(component %in% c("P0", "P1"), "all", mode),
         component = factor(component, levels = c("P0", "P1", "tree", "full"))) %>%
  summarise_trees(component, series, nSteps)
print(ggplot(d, aes(nSteps, median, colour = series, group = series)) +
        geom_ribbon(aes(ymin = low, ymax = high, fill = series), alpha = 0.15, colour = NA) +
        geom_line() + geom_point(aes(shape = extrapolated), size = 2, fill = "white") +
        scale_x_continuous(trans = "log2", breaks = 2^(10:14), labels = paste0("2^", 10:14)) + seconds_axis +
        facet_wrap(~component) +
        scale_colour_manual(values = mode_colours, name = "") + scale_fill_manual(values = mode_colours, name = "") +
        labs(title = "Run time against the number of steps (one type, shape 10, 100 tips)",
             subtitle = "Median and range of 5 trees. P0 and P1 do not depend on the tree. full = lifetimes + P0 + P1 + tree (annotated: no P1).",
             x = "number of steps", y = "seconds") + theme_runtime)

# page 2: tree size
d <- read_runtime("size") %>%
  filter(component %in% c("tree", "full")) %>%
  mutate(component = factor(component, levels = c("tree", "full")),
         steps = factor(paste0("2^", log2(nSteps)), levels = paste0("2^", c(10, 12, 14)))) %>%
  summarise_trees(component, steps, mode, tips)
print(ggplot(d, aes(tips, median, colour = mode, group = mode)) +
        geom_ribbon(aes(ymin = low, ymax = high, fill = mode), alpha = 0.15, colour = NA) +
        geom_line() + geom_point(aes(shape = extrapolated), size = 2, fill = "white") +
        scale_x_log10(breaks = c(10, 30, 100, 300, 1000, 3000), labels = plain) + seconds_axis +
        facet_grid(component ~ steps, labeller = label_both) +
        scale_colour_manual(values = mode_colours, name = "") + scale_fill_manual(values = mode_colours, name = "") +
        labs(title = "Run time against the tree size (one type, shape 10)",
             subtitle = "Median and range of 5 trees per size; the origin differs between trees. Open symbols: extrapolated.",
             x = "tips", y = "seconds") + theme_runtime)

# page 3: types; one type belongs to both graphs. Free parameters: lifetime, shape, death and sampling per type, and the
# asymmetric transition probabilities (the symmetric diagonal one is the remainder): chain types - 1, dense types (types - 1).
d <- read_runtime("ntypes")
d <- bind_rows(d %>% mutate(graph = ifelse(graph == "none", "chain", graph)),
               d %>% filter(graph == "none") %>% mutate(graph = "dense")) %>%
  filter((component %in% c("P0", "P1") & mode %in% c("all", "integrated")) |
           (component %in% c("tree", "full") & mode %in% c("integrated", "annotated"))) %>%
  mutate(what = factor(ifelse(component %in% c("tree", "full"), paste0(component, ", ", mode), component),
                       levels = c("P0", "P1", "tree, annotated", "tree, integrated", "full, annotated", "full, integrated")),
         graph = factor(graph, levels = c("chain", "dense"),
                        labels = c("chain: types - 1 transition parameters", "dense: types x (types - 1) transition parameters"))) %>%
  summarise_trees(graph, what, types)
print(ggplot(d, aes(types, median, colour = what, group = what)) +
        geom_ribbon(aes(ymin = low, ymax = high, fill = what), alpha = 0.15, colour = NA) +
        geom_line() + geom_point(aes(shape = extrapolated), size = 2, fill = "white") +
        scale_x_continuous(breaks = 1:10) + seconds_axis + facet_wrap(~graph) +
        labs(title = "Run time against the number of types (shape 10, 100 tips, 2^12 steps)",
             subtitle = "Median and range of 5 trees. chain: type i gives rise to type i + 1; dense: every pair. Open symbols: extrapolated.",
             x = "types", y = "seconds", colour = "", fill = "") + theme_runtime)

invisible(dev.off())
message("wrote ", file.path(dir, "runtime_plots.pdf"))
