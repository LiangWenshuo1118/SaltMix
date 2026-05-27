#!/usr/bin/env Rscript

Sys.setlocale("LC_ALL", "C")

experiment_dir <- Sys.getenv("EXPERIMENT_DIR", "outputs/2026_05_20_17_51")
processed_dir_name <- Sys.getenv("PROCESSED_DIR_NAME", "results_processed")

palette_contract <- c(
  neutral_black = "#272727",
  neutral_dark = "#4D4D4D",
  neutral_mid = "#767676",
  neutral_light = "#D8D8D8",
  non_phy = "#6A6A6A",
  baseline_dark = "#484878",
  baseline_mid = "#7884B4",
  baseline_soft = "#B4C0E4",
  ours_base = "#E4CCD8",
  signal_blue = "#3182BD",
  signal_teal = "#33B5A5",
  accent_red = "#D24B40",
  accent_orange = "#E28E2C",
  bg = "#FFFFFF"
)

base_model_order <- c("non_phy", "none", "0", "2", "4")
arrhenius_properties <- c("viscosity", "electrical_conductivity")
linear_properties <- c("density", "surface_tension")
model_axis_line1 <- c(non_phy = "Unconstrained", none = "", `0` = "N=0", `2` = "N=2", `4` = "N=4", arrhenius = "", linear = "")
model_axis_line2 <- c(non_phy = "", none = "", `0` = "CW+RK", `2` = "CW+RK", `4` = "CW+RK", arrhenius = "", linear = "")
model_axis_single <- c(non_phy = "", none = "CW", `0` = "", `2` = "", `4` = "",
                       arrhenius = "Arrhenius", linear = "Linear")
model_title_labels <- c(
  non_phy = "Unconstrained",
  none = "CW",
  `0` = "CW+RK N=0",
  `2` = "CW+RK N=2",
  `4` = "CW+RK N=4",
  arrhenius = "Arrhenius",
  linear = "Linear"
)
n_colours <- c(
  non_phy = palette_contract[["non_phy"]],
  none = palette_contract[["baseline_dark"]],
  `0` = palette_contract[["baseline_mid"]],
  `2` = palette_contract[["signal_teal"]],
  `4` = palette_contract[["accent_red"]],
  arrhenius = palette_contract[["accent_orange"]],
  linear = palette_contract[["accent_orange"]]
)

test_colour <- palette_contract[["signal_blue"]]
identity_line_colour <- palette_contract[["neutral_mid"]]
fit_line_colour <- palette_contract[["neutral_dark"]]

seed_colours <- c(
  `42` = "#484878",
  `43` = "#7884B4",
  `44` = "#33B5A5",
  `45` = "#E28E2C",
  `46` = "#D24B40"
)

seed_pch <- c(`42` = 21, `43` = 21, `44` = 21, `45` = 21, `46` = 21)
seed_labels <- c(`42` = "seed 1", `43` = "seed 2", `44` = "seed 3",
                 `45` = "seed 4", `46` = "seed 5")

properties <- data.frame(
  slug = c("density", "viscosity", "surface_tension", "electrical_conductivity"),
  title = c("Density", "Viscosity", "Surface tension", "Electrical conductivity"),
  metric_unit = c("g cm^-3", "ln(mPa s)", "mN m^-1", "ln(S cm^-1)"),
  value_factor = c(1, 1, 1, 1),
  stringsAsFactors = FALSE
)

model_order_for <- function(prop) {
  slug <- if (is.character(prop)) prop else prop$slug[[1]]
  if (slug %in% arrhenius_properties) {
    c(base_model_order, "arrhenius")
  } else if (slug %in% linear_properties) {
    c(base_model_order, "linear")
  } else {
    base_model_order
  }
}

best_rk_order_for <- function(data) {
  rk_order <- c("0", "2", "4")
  rk_metrics <- data$metrics[data$metrics$N %in% rk_order, ]
  if (nrow(rk_metrics) == 0) {
    stop("No CW+RK metrics available for selecting the displayed RK order.")
  }
  rk_metrics$rk_rank <- match(rk_metrics$N, rk_order)
  rk_metrics <- rk_metrics[order(rk_metrics$test_rmse_mean,
                                 rk_metrics$test_mae_mean,
                                 rk_metrics$rk_rank), ]
  rk_metrics$N[[1]]
}

correlation_model_order_for <- function(prop, data) {
  best_rk <- best_rk_order_for(data)
  slug <- if (is.character(prop)) prop else prop$slug[[1]]
  if (slug %in% arrhenius_properties) {
    c("non_phy", "none", best_rk, "arrhenius")
  } else if (slug %in% linear_properties) {
    c("non_phy", "none", best_rk, "linear")
  } else {
    c("non_phy", "none", best_rk)
  }
}

x_label_expr <- list(
  density = expression(Density~(g~cm^{-3})),
  viscosity = expression(ln(Viscosity / mPa~s)),
  surface_tension = expression(Surface~tension~(mN~m^{-1})),
  electrical_conductivity = expression(ln(Electrical~conductivity / S~cm^{-1}))
)

metric_unit_expr <- list(
  density = expression(g~cm^{-3}),
  surface_tension = expression(mN~m^{-1})
)

axis_label_for <- function(prop) {
  x_label_expr[[prop$slug]]
}

metric_unit_label_for <- function(prop) {
  metric_unit_expr[[prop$slug]]
}

point_value_columns <- function(data) {
  if ("target_log_value" %in% names(data)) {
    list(target = "target_log_value", prediction = "prediction_log_value")
  } else {
    list(target = "target_value", prediction = "prediction_value")
  }
}

mean_value_columns <- function(data) {
  if ("target_log_value" %in% names(data)) {
    list(
      target = "target_log_value",
      mean = "prediction_log_value_mean",
      sd = "prediction_log_value_sd"
    )
  } else {
    list(
      target = "target_value",
      mean = "prediction_value_mean",
      sd = "prediction_value_sd"
    )
  }
}

normalize_model_label <- function(values) {
  out <- tolower(trimws(values))
  out <- gsub("-", "_", out, fixed = FALSE)
  out[out %in% c("non_phy", "nonphysics", "no_physics", "no_phy")] <- "non_phy"
  out[out == "none"] <- "none"
  out[out == "phys_false"] <- "non_phy"
  out[out == "phys_true_rk_false"] <- "none"
  out[out == "phys_true_rk_true_n_0"] <- "0"
  out[out == "phys_true_rk_true_n_2"] <- "2"
  out[out == "phys_true_rk_true_n_4"] <- "4"
  out[out == "phys_true_arrhenius"] <- "arrhenius"
  out[out == "phys_true_linear"] <- "linear"
  out[out == "training_100pct_phys_false_rk_false_n_none"] <- "non_phy"
  out[out == "training_100pct_phys_true_rk_false_n_none"] <- "none"
  out[out == "training_100pct_phys_true_rk_true_n_0"] <- "0"
  out[out == "training_100pct_phys_true_rk_true_n_2"] <- "2"
  out[out == "training_100pct_phys_true_rk_true_n_4"] <- "4"
  out[out == "training_20pct_phys_false_rk_false_n_none"] <- "non_phy"
  out[out == "training_20pct_phys_true_rk_false_n_none"] <- "none"
  out[out == "training_20pct_phys_true_rk_true_n_0"] <- "0"
  out[out == "training_20pct_phys_true_rk_true_n_2"] <- "2"
  out[out == "training_20pct_phys_true_rk_true_n_4"] <- "4"
  out
}

read_property <- function(slug) {
  figure_root <- file.path(experiment_dir, slug)
  processed_root <- file.path(figure_root, processed_dir_name)
  metrics <- read.csv(file.path(processed_root, paste0(slug, "_metrics_by_n.csv")),
                      stringsAsFactors = FALSE)
  val <- read.csv(file.path(processed_root, paste0(slug, "_val_predictions_by_n_seed.csv")),
                  stringsAsFactors = FALSE)
  test <- read.csv(file.path(processed_root, paste0(slug, "_test_predictions_by_n_mean.csv")),
                   stringsAsFactors = FALSE)
  metrics$N <- normalize_model_label(metrics$N)
  val$N <- normalize_model_label(val$N)
  test$N <- normalize_model_label(test$N)
  list(root = processed_root, figure_root = figure_root, processed_root = processed_root,
       metrics = metrics, val = val, test = test)
}

with_pub_device <- function(filename, width_mm, height_mm, draw_fun, type, dpi = 1200) {
  width <- width_mm / 25.4
  height <- height_mm / 25.4

  if (type == "svg") {
    grDevices::svg(filename, width = width, height = height, family = "Arial",
                   pointsize = 10, bg = "white")
  } else if (type == "png") {
    grDevices::png(filename, width = width, height = height, units = "in",
                   res = dpi, type = "cairo", family = "Arial",
                   pointsize = 10, bg = "white")
  }
  on.exit(grDevices::dev.off(), add = TRUE)

  op <- par(no.readonly = TRUE)
  on.exit(par(op), add = TRUE)
  draw_fun()
}

save_pub_r <- function(base, width_mm, height_mm, draw_fun, dpi = 1200) {
  root <- dirname(base)
  stem <- basename(base)
  out_dirs <- c(svg = file.path(root, "fig_svg"),
                png = file.path(root, "fig_png"))
  dir.create(out_dirs[["svg"]], recursive = TRUE, showWarnings = FALSE)
  dir.create(out_dirs[["png"]], recursive = TRUE, showWarnings = FALSE)
  for (ext in c("svg", "png")) {
    filename <- file.path(out_dirs[[ext]], paste0(stem, ".", ext))
    with_pub_device(filename, width_mm, height_mm, draw_fun, ext, dpi = dpi)
  }
}

nature_par <- function(mar = c(3.15, 3.45, 1.35, 0.75), mgp = c(1.85, 0.48, 0)) {
  par(
    family = "Arial",
    las = 1,
    bty = "l",
    tcl = -0.23,
    lwd = 0.7,
    lend = "round",
    mar = mar,
    mgp = mgp,
    cex.axis = 0.82,
    cex.lab = 0.9,
    cex.main = 0.9,
    xaxs = "i",
    yaxs = "i"
  )
}

pad_range <- function(x, frac = 0.06) {
  rng <- range(x[is.finite(x)], na.rm = TRUE)
  if (!all(is.finite(rng))) return(c(0, 1))
  if (diff(rng) == 0) return(rng + c(-0.5, 0.5))
  rng + c(-1, 1) * diff(rng) * frac
}

format_metric <- function(x) {
  if (!is.finite(x)) return("NA")
  if (abs(x) >= 10) sprintf("%.1f", x) else sprintf("%.3f", x)
}

transform_values <- function(values, prop, is_log) {
  if (is_log) {
    values
  } else {
    values * prop$value_factor
  }
}

transform_spread <- function(values, prop, is_log) {
  if (is_log) values else values * prop$value_factor
}

panel_label <- function(label, x = par("usr")[1], y = par("usr")[4],
                        adj = c(-0.25, -0.2), cex = 1.25) {
  text(x, y, labels = label, xpd = NA, adj = adj,
       font = 2, cex = cex, col = palette_contract[["neutral_black"]])
}

draw_metric_panel <- function(metrics, measure, tag, prop) {
  n_order <- model_order_for(prop)
  x <- seq_along(n_order)
  idx <- match(n_order, metrics$N)
  if (any(is.na(idx))) {
    stop("Missing metrics for models: ", paste(n_order[is.na(idx)], collapse = ", "))
  }
  metrics <- metrics[idx, ]
  seed_ids <- c("42", "43", "44", "45", "46")
  val_matrix <- sapply(seed_ids, function(seed) metrics[[paste0("val_", measure, "_seed", seed)]])
  test_matrix <- sapply(seed_ids, function(seed) metrics[[paste0("test_", measure, "_seed", seed)]])
  is_log <- unique(metrics$metric_space)[1] == "log"
  if (!is_log) {
    val_matrix <- val_matrix * prop$value_factor
    test_matrix <- test_matrix * prop$value_factor
  }

  y_label <- toupper(measure)
  display_factor <- 1
  if (is_log) {
    y_label <- if (prop$slug == "electrical_conductivity") {
      as.expression(substitute(METRIC~ln(S~cm^{-1}),
        list(METRIC = as.name(y_label))
      ))
    } else {
      as.expression(substitute(METRIC~ln(mPa~s),
        list(METRIC = as.name(y_label))
      ))
    }
  } else {
    y_label <- as.expression(substitute(METRIC~UNIT,
      list(METRIC = as.name(y_label), UNIT = metric_unit_label_for(prop)[[1]])
    ))
  }
  val_matrix <- val_matrix * display_factor
  test_matrix <- test_matrix * display_factor

  val_mean <- rowMeans(val_matrix, na.rm = TRUE)
  test_mean <- rowMeans(test_matrix, na.rm = TRUE)
  val_sd <- apply(val_matrix, 1, stats::sd, na.rm = TRUE)
  test_sd <- apply(test_matrix, 1, stats::sd, na.rm = TRUE)
  y_lim <- c(0, max(c(val_mean + val_sd, test_mean + test_sd,
                     val_matrix, test_matrix), na.rm = TRUE) * 1.32)

  plot(x, rep(0, length(x)), type = "n", xaxt = "n", xlab = "", ylab = y_label,
       xlim = c(0.75, length(x) + 0.25), ylim = y_lim)
  axis(1, at = x, labels = FALSE, tick = FALSE)
  usr <- par("usr")
  label_y <- usr[3] - 0.08 * diff(usr[3:4])
  label_gap <- 0.08 * diff(usr[3:4])
  axis_label_cex <- if (length(n_order) > 5) 0.56 else 0.70
  text(x, rep(label_y, length(x)), labels = model_axis_line1[n_order], xpd = NA,
       adj = c(0.5, 0.5), cex = axis_label_cex, col = palette_contract[["neutral_black"]])
  text(x, rep(label_y - label_gap, length(x)), labels = model_axis_line2[n_order], xpd = NA,
       adj = c(0.5, 0.5), cex = axis_label_cex, col = palette_contract[["neutral_black"]])
  text(x, rep(label_y - 0.5 * label_gap, length(x)), labels = model_axis_single[n_order], xpd = NA,
       adj = c(0.5, 0.5), cex = axis_label_cex, col = palette_contract[["neutral_black"]])
  abline(h = pretty(y_lim, 4), col = "#EEEEEE", lwd = 0.42)

  split_offsets <- c(validation = -0.18, test = 0.18)
  bar_w <- 0.26
  for (i in seq_along(x)) {
    val_x <- x[i] + split_offsets[["validation"]]
    test_x <- x[i] + split_offsets[["test"]]
    rect(val_x - bar_w / 2, 0, val_x + bar_w / 2, val_mean[i],
         col = adjustcolor(palette_contract[["signal_blue"]], alpha.f = 0.66),
         border = palette_contract[["signal_blue"]], lwd = 0.45)
    rect(test_x - bar_w / 2, 0, test_x + bar_w / 2, test_mean[i],
         col = adjustcolor(palette_contract[["accent_red"]], alpha.f = 0.66),
         border = palette_contract[["accent_red"]], lwd = 0.45)
    arrows(val_x, val_mean[i] - val_sd[i], val_x, val_mean[i] + val_sd[i],
           angle = 90, code = 3, length = 0.025, lwd = 0.85,
           col = palette_contract[["neutral_dark"]])
    arrows(test_x, test_mean[i] - test_sd[i], test_x, test_mean[i] + test_sd[i],
           angle = 90, code = 3, length = 0.025, lwd = 0.85,
           col = palette_contract[["neutral_dark"]])
    points(rep(val_x, length(seed_ids)), val_matrix[i, ], pch = 21, bg = "white",
           col = palette_contract[["neutral_black"]], cex = 0.45, lwd = 0.55)
    points(rep(test_x, length(seed_ids)), test_matrix[i, ], pch = 21, bg = "white",
           col = palette_contract[["neutral_black"]], cex = 0.45, lwd = 0.55)
  }

  usr <- par("usr")
  legend_x <- usr[1] + 0.56 * diff(usr[1:2])
  legend_y <- usr[4] - 0.02 * diff(usr[3:4])
  legend(legend_x, legend_y, legend = c("Validation", "Test"), bty = "n",
         fill = adjustcolor(c(palette_contract[["signal_blue"]],
                              palette_contract[["accent_red"]]), alpha.f = 0.66),
         border = c(palette_contract[["signal_blue"]], palette_contract[["accent_red"]]),
         cex = 0.72, y.intersp = 0.82, x.intersp = 0.62)
  panel_label(tag, cex = 1.02)
}

draw_metrics_figure <- function(prop, data) {
  n_order <- model_order_for(prop)
  save_pub_r(
    file.path(data$root, paste0(prop$slug, "_rmse_mae_by_n")),
    if (length(n_order) > 5) 183 else 120, 62,
    function() {
      par(mfrow = c(1, 2), oma = c(0.8, 0, 1.1, 0))
      nature_par()
      draw_metric_panel(data$metrics, "rmse", "a", prop)
      nature_par()
      draw_metric_panel(data$metrics, "mae", "b", prop)
      mtext(paste0(prop$title, ": model performance"),
            outer = TRUE, side = 3, line = 0.05, adj = 0,
            font = 2, cex = 0.92, col = palette_contract[["neutral_black"]])
    }
  )
}

scatter_stats <- function(target, prediction) {
  keep <- is.finite(target) & is.finite(prediction)
  target <- target[keep]
  prediction <- prediction[keep]
  residual <- prediction - target
  r <- suppressWarnings(cor(target, prediction))
  c(
    n = length(target),
    r = ifelse(is.finite(r), r, NA_real_),
    rmse = sqrt(mean(residual^2)),
    mae = mean(abs(residual))
  )
}

draw_scatter <- function(target, prediction, spread, title, axis_label, colour,
                         tag = NULL, compact = FALSE, group = NULL,
                         show_group_legend = FALSE, show_axis_labels = TRUE) {
  keep <- is.finite(target) & is.finite(prediction)
  target <- target[keep]
  prediction <- prediction[keep]
  if (!is.null(spread)) spread <- spread[keep]
  if (!is.null(group)) group <- as.character(group[keep])

  lim <- pad_range(c(target, prediction), 0.07)
  x_axis_label <- if (show_axis_labels) axis_label else ""
  y_axis_label <- if (show_axis_labels) "Predicted" else ""
  plot(target, prediction, type = "n", xlim = lim, ylim = lim,
       xlab = x_axis_label, ylab = y_axis_label)
  abline(h = pretty(lim, 5), v = pretty(lim, 5), col = "#F0F0F0", lwd = 0.45)
  abline(0, 1, col = identity_line_colour, lwd = 1.05)

  if (!is.null(spread)) {
    arrows(target, prediction - spread, target, prediction + spread,
           angle = 90, code = 3, length = 0.015, lwd = 0.35,
           col = adjustcolor(colour, alpha.f = 0.28))
  }

  if (is.null(group)) {
    points(target, prediction, pch = 21, cex = if (compact) 0.54 else 0.58,
           bg = adjustcolor(colour, alpha.f = if (compact) 0.35 else 0.42),
           col = adjustcolor(palette_contract[["neutral_black"]],
                             alpha.f = if (compact) 0.16 else 0.22),
           lwd = 0.3)
  } else {
    groups <- intersect(names(seed_colours), unique(group))
    if (length(groups) == 0) groups <- sort(unique(group))
    for (g in groups) {
      idx <- group == g
      bg_col <- if (g %in% names(seed_colours)) seed_colours[[g]] else colour
      point_shape <- if (g %in% names(seed_pch)) seed_pch[[g]] else 21
      points(target[idx], prediction[idx], pch = point_shape,
             cex = if (compact) 0.54 else 0.62,
             bg = "white",
             col = adjustcolor(bg_col, alpha.f = if (compact) 0.68 else 0.82),
             lwd = if (compact) 0.42 else 0.55)
    }
    if (show_group_legend) {
      legend_groups <- intersect(names(seed_colours), sort(unique(group)))
      legend("bottomright", legend = seed_labels[legend_groups],
             pch = 21, pt.bg = "white",
             col = seed_colours[legend_groups],
             bty = "n", cex = if (compact) 0.70 else 0.72,
             pt.cex = if (compact) 0.88 else 0.9,
             y.intersp = if (compact) 0.70 else 0.82,
             x.intersp = if (compact) 0.50 else 0.65)
    }
  }

  fit <- try(stats::lm(prediction ~ target), silent = TRUE)
  if (!inherits(fit, "try-error")) {
    abline(fit, col = fit_line_colour, lwd = 1.25)
  }

  st <- scatter_stats(target, prediction)
  usr <- par("usr")
  stats_labels <- as.expression(c(
    substitute(italic(r) == VALUE, list(VALUE = format_metric(st[["r"]]))),
    substitute(RMSE == VALUE, list(VALUE = format_metric(st[["rmse"]]))),
    substitute(MAE == VALUE, list(VALUE = format_metric(st[["mae"]]))),
    substitute(n == VALUE, list(VALUE = st[["n"]]))
  ))
  text(usr[1] + 0.04 * diff(usr[1:2]),
       usr[4] - c(0.07, 0.16, 0.25, 0.34) * diff(usr[3:4]),
       labels = stats_labels,
       adj = c(0, 1), cex = if (compact) 1.02 else 0.82,
       col = palette_contract[["neutral_dark"]])
  if (!is.null(title) && nzchar(title)) {
    title(main = title, adj = if (compact) 0.16 else 0, line = 0.25,
          cex.main = if (compact) 1.04 else par("cex.main"),
          col.main = palette_contract[["neutral_black"]])
  }
  if (!is.null(tag)) {
    if (compact) {
      usr <- par("usr")
      panel_label(tag,
                  x = usr[1] - 0.07 * diff(usr[1:2]),
                  y = usr[4] + 0.10 * diff(usr[3:4]),
                  adj = c(0, 1), cex = 1.42)
    } else {
      panel_label(tag)
    }
  }
}

draw_grid_figure <- function(prop, data) {
  n_order <- correlation_model_order_for(prop, data)
  n_models <- length(n_order)
  save_pub_r(
    file.path(data$root, paste0(prop$slug, "_correlation_grid")),
    183, 104,
    function() {
      par(mfrow = c(2, n_models), oma = c(2.0, 2.8, 0.1, 0))
      tags <- letters[seq_len(2 * n_models)]
      tag_i <- 1
      for (split in c("val", "test")) {
        for (n_value in n_order) {
          nature_par(mar = c(2.35, 2.18, 1.75, 0.52),
                     mgp = c(1.70, 0.48, 0))
          par(cex.axis = 0.92, cex.lab = 0.98, cex.main = 1.00)
          if (split == "val") {
            df <- data$val[data$val$N == n_value, ]
            cols <- point_value_columns(df)
            is_log <- grepl("_log_", cols$target)
            target <- transform_values(df[[cols$target]], prop, is_log)
            prediction <- transform_values(df[[cols$prediction]], prop, is_log)
            spread <- NULL
            group <- df$seed
            split_label <- "Validation"
          } else {
            df <- data$test[data$test$N == n_value, ]
            cols <- mean_value_columns(df)
            is_log <- grepl("_log_", cols$target)
            target <- transform_values(df[[cols$target]], prop, is_log)
            prediction <- transform_values(df[[cols$mean]], prop, is_log)
            spread <- transform_spread(df[[cols$sd]], prop, is_log)
            group <- NULL
            split_label <- "Test"
          }
          draw_scatter(
            target, prediction, spread,
            paste0(split_label, " | ", model_title_labels[[n_value]]),
            axis_label_for(prop),
            if (split == "test") test_colour else n_colours[[n_value]],
            tags[[tag_i]], compact = TRUE,
            group = group, show_group_legend = split == "val" && n_value == "non_phy",
            show_axis_labels = FALSE
          )
          tag_i <- tag_i + 1
        }
      }
      mtext(axis_label_for(prop), outer = TRUE, side = 1, line = 0.7,
            cex = 1.12, col = palette_contract[["neutral_black"]])
      mtext("Predicted", outer = TRUE, side = 2, line = 0.75, las = 0,
            cex = 1.12, col = palette_contract[["neutral_black"]])
    },
    dpi = 600
  )
}

write_source_data <- function(prop, data) {
  n_order <- model_order_for(prop)
  metrics <- data$metrics[match(n_order, data$metrics$N), ]
  if (any(is.na(metrics$N))) {
    stop("Missing source metrics for models: ", paste(n_order[is.na(metrics$N)], collapse = ", "))
  }
  out <- metrics[, c("N", "metric_space", "n_seeds",
                     "val_rmse_mean", "val_rmse_sd", "test_rmse_mean",
                     "test_rmse_sd", "val_mae_mean", "val_mae_sd",
                     "test_mae_mean", "test_mae_sd")]
  out$plotted_unit <- prop$metric_unit
  if (prop$metric_unit %in% names(metric_unit_expr)) {
    metric_cols <- grep("_(rmse|mae)_(mean|sd)$", names(out), value = TRUE)
    out[metric_cols] <- lapply(out[metric_cols], function(x) x * prop$value_factor)
  }
  write.csv(out,
            file.path(data$root, paste0(prop$slug, "_source_metrics.csv")),
            row.names = FALSE)
}

main <- function() {
  for (i in seq_len(nrow(properties))) {
    prop <- properties[i, ]
    data <- read_property(prop$slug)
    write_source_data(prop, data)
    draw_metrics_figure(prop, data)
    draw_grid_figure(prop, data)
  }
}

main()
