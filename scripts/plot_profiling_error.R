# Plots and tabulates the discretisation error written by RuntimeAndErrorProfiling (testErrorVsSampling*)
#
# Usage (from the repository root): Rscript scripts/plot_profiling_error.R
#
# Reads profiling/results/runtime_and_error/error_{rho_origin,rho_taxa}.csv (and the trees in profiling/trees for the
# total branch length) and writes profiling/results/runtime_and_error/error_plots.pdf and prints Markdown tables.
# The error of a log likelihood is its value minus the reference, the extrapolated value of the two finest grids (first
# order in the step). Open symbols: the convergence ratio of the reference is below 3.5 (4 = first order), so the
# reference is still too far from the limit and the error is a lower bound (small sampling proportion, long origin).
#   page 1  absolute error against the number of steps, per sampling proportion rho, mode, and design
#           (fixed origin 13: the number of tips follows rho; 100 tips: the origin follows rho)
#   page 2  error at 2^14 steps per unit of total branch length against rho: both designs on one curve
# Modes: integrated (hidden divisions integrated out), approx (series approximation), annotated.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(ape)
})

dir <- "profiling/results/runtime_and_error"
designs <- c(rho_origin = "fixed origin 13", rho_taxa = "100 tips")

tree_length <- function(experiment) {
  t <- read.delim(file.path("profiling/trees", paste0("trees_", experiment, ".tsv")), quote = "", stringsAsFactors = FALSE)
  t$length <- vapply(t$newick_collapsed, function(s) sum(read.tree(text = gsub("\\[&type=[0-9]+\\]", "", s))$edge.length), 0)
  t[, c("config", "tree", "length")]
}

errors <- bind_rows(lapply(names(designs), function(e) {
  x <- read.csv(file.path(dir, paste0("error_", e, ".csv")), na.strings = "NA")
  x$error <- suppressWarnings(as.numeric(x$error))
  x$ratio <- suppressWarnings(as.numeric(x$ratio))
  x %>% left_join(tree_length(e), by = c("config", "tree")) %>%
    mutate(design = designs[[e]], rho = factor(rho))
})) %>%
  filter(!is.na(error), error != 0 | nSteps != finestGrid) %>%
  mutate(abs_error = abs(error), reliable = ratio >= 3.5,
         design = factor(design, levels = designs), mode = factor(mode, levels = c("integrated", "approx", "annotated")))

# the finest grid has an error of about 1/3 of the one before it by construction of the reference; keep all rows but
# the error of the finest grid carries the extrapolation (shown as is)
plain <- function(x) format(x, scientific = FALSE, drop0trailing = TRUE, trim = TRUE)
summary_rows <- errors %>%
  group_by(design, mode, rho, nSteps) %>%
  summarise(median = median(abs_error), low = min(abs_error), high = max(abs_error), reliable = all(reliable), n = n(), .groups = "drop") %>%
  filter(median > 0)

pdf(file.path(dir, "error_plots.pdf"), width = 11, height = 7.5)
print(ggplot(summary_rows, aes(nSteps, median, colour = rho, group = rho)) +
        geom_ribbon(aes(ymin = low, ymax = high, fill = rho), alpha = 0.12, colour = NA) +
        geom_line() + geom_point(aes(shape = reliable), size = 2, fill = "white") +
        scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), guide = "none") +
        scale_x_continuous(trans = "log2", breaks = 2^seq(10, 20, 2), labels = paste0("2^", seq(10, 20, 2))) +
        scale_y_log10(breaks = 10^(-6:2), labels = plain) +
        facet_grid(mode ~ design) +
        scale_colour_brewer(palette = "Dark2", name = "sampling proportion") +
        scale_fill_brewer(palette = "Dark2", name = "sampling proportion") +
        labs(title = "Discretisation error of the log likelihood against the number of steps",
             subtitle = "Median and range of 5 trees (one type, shape 10). Error = value - extrapolated reference. Open symbols: reference not in the first-order regime (error is a lower bound).\nIntegrated: up to 2^16 steps, approx and annotated up to 2^20; trees whose integrated value underflows (bug 13) are left out.",
             x = "number of steps", y = "|error| (log likelihood)") +
        theme_bw(base_size = 11) + theme(legend.position = "bottom", plot.subtitle = element_text(size = 8),
                                         panel.grid.minor = element_blank()))

per_length <- errors %>% filter(nSteps == 16384) %>%
  mutate(per_length = abs_error / length, rho = as.numeric(as.character(rho))) %>%
  group_by(design, mode, rho) %>%
  summarise(median = median(per_length), low = min(per_length), high = max(per_length), reliable = all(reliable), .groups = "drop")
print(ggplot(per_length, aes(rho, median, colour = design, shape = design)) +
        geom_line() + geom_point(size = 2.5) +
        geom_errorbar(aes(ymin = low, ymax = high), width = 0.1, alpha = 0.5) +
        facet_wrap(~mode) +
        scale_x_log10(breaks = c(0.001, 0.01, 0.1, 1), labels = plain) +
        scale_y_log10(breaks = 10^(-7:0), labels = plain) +
        labs(title = "Error at 2^14 steps per unit of total branch length",
             subtitle = "Median and range of 5 trees. The two designs give the same curve: the error is proportional to the tree length and rises steeply as the sampling proportion falls.",
             x = "sampling proportion", y = "|error| / total branch length", colour = "", shape = "") +
        theme_bw(base_size = 11) + theme(legend.position = "bottom", plot.subtitle = element_text(size = 8),
                                         panel.grid.minor = element_blank()))
invisible(dev.off())
message("wrote ", file.path(dir, "error_plots.pdf"))

# tables: median |error| at 2^10, 2^12, 2^14, 2^16 and the number of steps for an error below 0.1 (first order: error * nSteps is constant)
cat("\n#### Median absolute error of the log likelihood (nSteps 2^10 / 2^12 / 2^14 / 2^16)\n\n")
tab <- summary_rows %>% filter(nSteps %in% 2^c(10, 12, 14, 16)) %>%
  mutate(cell = ifelse(reliable, sprintf("%.3g", median), sprintf("%.3g*", median))) %>%
  select(design, mode, rho, nSteps, cell) %>%
  tidyr::pivot_wider(names_from = nSteps, values_from = cell, names_prefix = "2^")
names(tab) <- sub("2\\^1024$", "2^10", sub("2\\^4096$", "2^12", sub("2\\^16384$", "2^14", sub("2\\^65536$", "2^16", names(tab)))))
cat(knitr::kable(tab %>% arrange(design, mode, rho), format = "pipe"), sep = "\n")
cat("\n(* reference not in the first-order regime: lower bound)\n")

cat("\n#### Steps needed for |error| below 0.1 (first order: nSteps at 2^14 x error / 0.1)\n\n")
need <- summary_rows %>% filter(nSteps == 16384) %>%
  mutate(needed = 16384 * median / 0.1,
         needed = ifelse(needed <= 16384, "2^14 or fewer", sprintf("%s2^%.1f", ifelse(reliable, "", ">"), log2(needed)))) %>%
  select(design, mode, rho, needed) %>% tidyr::pivot_wider(names_from = rho, values_from = needed, names_prefix = "rho ")
cat(knitr::kable(need %>% arrange(design, mode), format = "pipe"), sep = "\n")
