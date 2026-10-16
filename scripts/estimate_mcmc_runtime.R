# Estimates the run time of 1 million MCMC samples from the run times of RuntimeAndErrorProfiling
#
# Usage (from the repository root): Rscript scripts/estimate_mcmc_runtime.R
#
# Also writes profiling/results/runtime_and_error/mcmc_runtime_estimate.pdf:
# one panel per experiment (steps, tree size, types), the time of 10^6 samples on a log axis
# (line: 20% parameter proposals; ribbon: from tree proposals only to parameter proposals only),
# with reference lines at 1 hour, 1 day, 1 week, 1 month and 1 year.
#
# Reads profiling/results/runtime_and_error/runtime_{nsteps,size,ntypes}.csv (median over the 5 trees) and prints Markdown
# tables. One sample = one proposal = one evaluation of the tree prior. A proposal that changes a model parameter
# (lifetime, death, sampling, transitions; P0 and P1 are solved again) costs the "full" time, a proposal that only changes
# the tree costs the "tree" time. Three mixes: only tree proposals, 20% parameter proposals, only parameter proposals.
# Excluded: other posterior components, the cost of the operators themselves, store/restore. Single thread.

suppressPackageStartupMessages(library(dplyr))
dir <- "profiling/results/runtime_and_error"
mix <- c("tree proposals only" = 0, "20% parameter proposals" = 0.2, "parameter proposals only" = 1)

format_time <- function(seconds) {
  hours <- seconds / 3600
  ifelse(hours < 10, sprintf("%.2g h", hours),
         ifelse(hours < 48, sprintf("%.0f h", hours),
         ifelse(hours < 24 * 14, sprintf("%.1f d", hours / 24), sprintf("%.1f weeks", hours / 24 / 7))))
}

estimate <- function(d, keys) {
  d %>%
    filter(component %in% c("tree", "full")) %>%
    group_by(across(all_of(c(keys, "component")))) %>%
    summarise(seconds = median(seconds), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = component, values_from = seconds) %>%
    rowwise() %>%
    mutate(across(everything(), identity)) %>%
    ungroup() %>%
    (function(w) {
      for (name in names(mix)) w[[name]] <- format_time(1e6 * (mix[[name]] * w$full + (1 - mix[[name]]) * w$tree))
      w$`per full evaluation (s)` <- signif(w$full, 2)
      w[, c(keys, "per full evaluation (s)", names(mix))]
    })
}

show <- function(title, d) {
  cat("\n####", title, "\n\n")
  cat(knitr::kable(d, format = "pipe"), sep = "\n")
}

n <- read.csv(file.path(dir, "runtime_nsteps.csv")) %>% filter(mode != "all")
show("Number of steps (one type, shape 10, 100 tips)",
     estimate(n, c("mode", "nSteps")) %>% arrange(mode, nSteps))

s <- read.csv(file.path(dir, "runtime_size.csv")) %>% filter(nSteps == 4096, mode != "all")
show("Tree size (one type, shape 10, 2^12 steps)", estimate(s, c("mode", "tips")) %>% arrange(mode, tips))

t <- read.csv(file.path(dir, "runtime_ntypes.csv")) %>% filter(mode %in% c("integrated", "annotated"))
show("Number of types (shape 10, 100 tips, 2^12 steps)", estimate(t, c("mode", "graph", "types")) %>% arrange(mode, graph, types))


# ------------------------------------------------------------------ plot
suppressPackageStartupMessages({ library(ggplot2); library(patchwork) })

hours <- function(d, keys) {
  d %>%
    filter(component %in% c("tree", "full")) %>%
    group_by(across(all_of(c(keys, "component")))) %>%
    summarise(seconds = median(seconds), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = component, values_from = seconds) %>%
    mutate(tree_only = 1e6 * tree / 3600, mixed = 1e6 * (0.2 * full + 0.8 * tree) / 3600,
           parameters_only = 1e6 * full / 3600)
}

mode_colours <- c(integrated = "#d95f02", approx = "#1b9e77", annotated = "#7570b3")
reference <- data.frame(hours = c(1, 24, 168, 720, 8760), label = c("1 hour", "1 day", "1 week", "1 month", "1 year"))
plot_hours <- function(d, x, panel_title, x_scale, show_labels = FALSE, show_line = FALSE) {
  ggplot(d, aes(.data[[x]], mixed, colour = mode, group = interaction(mode, line))) +
    geom_hline(data = reference, aes(yintercept = hours), colour = "grey70", linetype = "dotted") +
    {if (show_labels) geom_text(data = transform(reference, xvalue = min(d[[x]])), aes(x = xvalue, y = hours, label = label),
                                inherit.aes = FALSE, hjust = 0, vjust = -0.5, size = 3, colour = "grey40")} +
    geom_ribbon(aes(ymin = tree_only, ymax = parameters_only, fill = mode), alpha = 0.18, colour = NA) +
    geom_line(aes(linetype = line)) +
    geom_point(aes(shape = invalid), size = 2, fill = "white") +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 21), guide = "none") +
    scale_linetype_manual(values = c(all = "solid", chain = "solid", dense = "dashed"), name = "graph",
                          breaks = c("chain", "dense"), guide = if (show_line) "legend" else "none") +
    x_scale +
    scale_y_log10(breaks = reference$hours, labels = reference$label, limits = c(0.01, 2e5)) +
    scale_colour_manual(values = mode_colours, limits = names(mode_colours), name = "") +
    scale_fill_manual(values = mode_colours, limits = names(mode_colours), name = "") +
    labs(title = panel_title, x = x, y = "time of 10^6 samples") +
    theme_bw(base_size = 11) +
    theme(plot.title = element_text(size = 10), panel.grid.minor = element_blank())
}
no_y <- list(theme(axis.text.y = element_blank(), axis.title.y = element_blank(), axis.ticks.y = element_blank()),
             guides(colour = "none", fill = "none")) # the legend of the first panel is shared

d <- hours(n, c("mode", "nSteps")) %>% mutate(invalid = FALSE, line = "all")
p1 <- plot_hours(d, "nSteps", "number of steps (100 tips, 1 type)",
                 scale_x_continuous(trans = "log2", breaks = 2^(10:14), labels = paste0("2^", 10:14)), show_labels = TRUE)
d <- hours(s, c("mode", "tips")) %>% mutate(invalid = mode != "annotated" & tips >= 1000, line = "all")
p2 <- plot_hours(d, "tips", "tree size (1 type, 2^12 steps)",
                 scale_x_log10(breaks = c(10, 30, 100, 300, 1000, 3000), labels = c(10, 30, 100, 300, 1000, 3000))) + no_y
d <- hours(t, c("mode", "graph", "types")) %>% mutate(invalid = FALSE)
d <- bind_rows(d %>% filter(graph != "none"), d %>% filter(graph == "none") %>% mutate(graph = "chain"),
               d %>% filter(graph == "none") %>% mutate(graph = "dense")) %>% mutate(line = graph)
p3 <- plot_hours(d, "types", "number of types (100 tips, 2^12 steps)",
                 scale_x_continuous(breaks = 1:10), show_line = TRUE) + no_y

pdf(file.path(dir, "mcmc_runtime_estimate.pdf"), width = 14, height = 6)
print((p1 | p2 | p3) + patchwork::plot_layout(guides = "collect") +
        patchwork::plot_annotation(
          title = "Time of one million MCMC samples (shape 10, integrated, approx and annotated likelihood)",
          subtitle = paste0("Line: 20% parameter proposals, 80% tree proposals; ribbon: from tree proposals only to parameter proposals only. ",
                            "Single thread, Apple M3. Log time axis, the same in all panels.\n",
                            "Open symbols (tree size 1000 and 3000): the integrated and approx log likelihood underflows (bug 13). ",
                            "Types: solid chain graph, dashed dense graph (they almost coincide).")) &
        theme(legend.position = "bottom", plot.subtitle = element_text(size = 8)))
invisible(dev.off())
message("wrote ", file.path(dir, "mcmc_runtime_estimate.pdf"))
