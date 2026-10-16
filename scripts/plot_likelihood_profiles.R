# Plots the profile likelihood curves written by ProfileLikelihoodTest
#
# Usage (from the repository root): Rscript scripts/plot_likelihood_profiles.R
#
# Reads profiling/results/likelihood_profiles/profile_<config>_<variant>.csv (columns parameter, value, logL, truth) and writes two files,
# profiling/results/likelihood_profiles/profiles_relative.pdf (relative log likelihoods) and profiles_absolute.pdf (absolute log likelihoods),
# each with one page per configuration. A page has one row per variant (annotated trees: hidden divisions given;
# integrated: hidden divisions integrated out) and one column per parameter, each panel with its own axes.
# For one type, the curves of the series approximation (variant approx) are drawn over the integrated curves in
# light blue. The curve is the pooled log likelihood, relative to the maximum of its own curve in profiles_relative.pdf
# and absolute in profiles_absolute.pdf; all grid points are shown. The dashed line is the true value;
# in profiles_relative.pdf the dotted line is 4.5 below the maximum (3 standard errors): the curve has to be above
# it at the true value. Lifetime, death and sampling are factors on the true values (1 = truth).

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

files <- list.files("profiling/results/likelihood_profiles", pattern = "^profile_.*\\.csv$", full.names = TRUE)
curves <- bind_rows(lapply(files, function(file) {
  name <- sub("^profile_", "", sub("\\.csv$", "", basename(file)))
  variant <- sub("^.*_", "", name)
  read.csv(file) %>%
    mutate(config = sub("_[a-z]+$", "", name),
           variant = variant,
           truth = as.logical(truth))
})) %>%
  group_by(config, variant, parameter) %>%
  mutate(relative = logL - max(logL)) %>%
  ungroup() %>%
  mutate(method = factor(ifelse(variant == "approx", "series approximation", "Picard iteration"),
                         levels = c("Picard iteration", "series approximation")),
         variant = factor(ifelse(variant == "approx", "integrated", variant), levels = c("annotated", "integrated")),
         parameter = factor(parameter, levels = c("lifetime", "death", "sampling", "shape", "transition")))

describe <- function(config) {
  m <- regmatches(config, regexec("^adb_([12])type_shape([0-9]+)$", config))[[1]]
  paste0(m[2], if (m[2] == "1") " type" else " types", ", true shape ", m[3])
}

plot_profiles <- function(file, is_relative) {
  pdf(file, width = 11, height = 6)
  for (config in sort(unique(curves$config))) {
    data <- filter(curves, config == !!config) %>% mutate(y = if (is_relative) .data$relative else .data$logL)
    plot <- ggplot(data, aes(value, y)) +
      geom_vline(data = filter(data, truth), aes(xintercept = value), linetype = "dashed") +
      geom_line(data = filter(data, method == "Picard iteration")) +
      geom_point(data = filter(data, method == "Picard iteration"), size = 1) +
      geom_line(data = filter(data, method == "series approximation"), aes(colour = method), linewidth = 1.2, alpha = 0.7) +
      scale_colour_manual(values = c("series approximation" = "lightblue"), name = NULL) +
      facet_grid(variant ~ parameter, scales = "free", labeller = labeller(variant = c(
        annotated = "annotated trees", integrated = "integrated (collapsed trees)"))) +
      labs(title = paste("Profile likelihood:", describe(config)),
           subtitle = if (is_relative) "Pooled log likelihood relative to its maximum. Dashed: true value. Dotted: 4.5 below the maximum (3 standard errors)."
                      else "Pooled log likelihood (absolute). Dashed: true value.",
           x = "parameter value (factor on the true values for lifetime, death and sampling)",
           y = if (is_relative) "log likelihood relative to the maximum" else "log likelihood") +
      theme_bw() +
      theme(plot.subtitle = element_text(size = 8), legend.position = "bottom")
    if (is_relative) plot <- plot + geom_hline(yintercept = -4.5, linetype = "dotted")
    suppressWarnings(print(plot))
  }
  invisible(dev.off())
}

plot_profiles("profiling/results/likelihood_profiles/profiles_relative.pdf", is_relative = TRUE)
plot_profiles("profiling/results/likelihood_profiles/profiles_absolute.pdf", is_relative = FALSE)
