#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
experiment_dir <- Sys.getenv("EXPERIMENT_DIR", "outputs/2026_05_20_17_51")
processed_dir_name <- Sys.getenv("PROCESSED_DIR_NAME", "results_processed")

model_labels <- c(
  non_phy = "Unconstrained model",
  none = "CW model",
  `0` = "CW+RK model N=0",
  `2` = "CW+RK model N=2",
  `4` = "CW+RK model N=4",
  linear = "Linear model",
  arrhenius = "Arrhenius model"
)

property_defaults <- data.frame(
  slug = c("density", "surface_tension", "viscosity", "electrical_conductivity"),
  title = c("Density", "Surface tension", "Viscosity", "Electrical conductivity"),
  model = c("0", "linear", "arrhenius", "arrhenius"),
  stringsAsFactors = FALSE
)

palette_contract <- c(
  points = "#2C7FB8",
  points_fill = "#6BAED6",
  identity = "#555555",
  fit = "#222222",
  grid = "#EEEEEE",
  text = "#2B2B2B"
)

split_csv_values <- function(value) {
  values <- strsplit(value, ",", fixed = TRUE)[[1]]
  trimws(values[nzchar(trimws(values))])
}

normalize_model_id <- function(x) {
  x <- tolower(trimws(x))
  x <- gsub("-", "_", x)
  x[x %in% c("nonphysics", "no_physics", "no_phy")] <- "non_phy"
  x[x == "phys_false"] <- "non_phy"
  x[x == "phys_true_rk_false"] <- "none"
  x[x == "phys_true_rk_true_n_0"] <- "0"
  x[x == "phys_true_rk_true_n_2"] <- "2"
  x[x == "phys_true_rk_true_n_4"] <- "4"
  x[x == "phys_true_linear"] <- "linear"
  x[x == "phys_true_arrhenius"] <- "arrhenius"
  x
}

safe_model_name <- function(x) {
  out <- if (x %in% names(model_labels)) model_labels[[x]] else x
  out <- gsub("[^A-Za-z0-9]+", "_", out)
  out <- gsub("^_|_$", "", out)
  tolower(out)
}

display_model_name <- function(x) {
  if (x %in% names(model_labels)) model_labels[[x]] else x
}

target_axis_label <- function(property_slug) {
  switch(
    property_slug,
    density = expression(Target~density~(g~cm^{-3})),
    surface_tension = expression(Target~surface~tension~(mN~m^{-1})),
    viscosity = expression(Target~ln(viscosity / mPa~s)),
    electrical_conductivity = expression(Target~ln(electrical~conductivity / S~cm^{-1})),
    expression(Target)
  )
}

prediction_axis_label <- function(property_slug) {
  switch(
    property_slug,
    density = expression(Predicted~density~(g~cm^{-3})),
    surface_tension = expression(Predicted~surface~tension~(mN~m^{-1})),
    viscosity = expression(Predicted~ln(viscosity / mPa~s)),
    electrical_conductivity = expression(Predicted~ln(electrical~conductivity / S~cm^{-1})),
    expression(Predicted)
  )
}

requested_properties <- function(args) {
  property_env <- Sys.getenv("PROPERTY", "")
  properties_env <- Sys.getenv("PROPERTIES", "")
  model_env <- Sys.getenv("MODEL", "")

  if (length(args) == 1 && normalize_model_id(args[[1]]) %in% names(model_labels)) {
    cfg <- property_defaults[property_defaults$slug == "density", ]
    cfg$model <- normalize_model_id(args[[1]])
    return(cfg)
  }

  if (length(args) > 0) {
    requested <- split_csv_values(paste(args, collapse = ","))
  } else if (nzchar(property_env)) {
    requested <- split_csv_values(property_env)
  } else if (nzchar(properties_env)) {
    requested <- split_csv_values(properties_env)
  } else {
    requested <- "all"
  }

  if (length(requested) == 1 && identical(tolower(requested[[1]]), "all")) {
    cfg <- property_defaults
  } else {
    requested <- tolower(trimws(requested))
    cfg <- property_defaults[property_defaults$slug %in% requested, ]
    missing <- setdiff(requested, cfg$slug)
    if (length(missing)) {
      stop("Unknown property requested: ", paste(missing, collapse = ", "),
           ". Available properties: ", paste(property_defaults$slug, collapse = ", "))
    }
    cfg <- cfg[match(requested, cfg$slug), ]
  }

  if (nzchar(model_env)) {
    if (nrow(cfg) != 1) {
      stop("MODEL can only override a single requested property.")
    }
    cfg$model <- normalize_model_id(model_env)
  }
  cfg
}

scatter_r <- function(target, prediction) {
  keep <- is.finite(target) & is.finite(prediction)
  target <- target[keep]
  prediction <- prediction[keep]
  r <- suppressWarnings(stats::cor(target, prediction))
  if (is.finite(r)) r else NA_real_
}

fmt <- function(x) {
  if (!is.finite(x)) {
    return("NA")
  }
  formatC(x, digits = 3, format = "fg", flag = "#")
}

chemical_formula_expr <- function(formula) {
  chars <- strsplit(formula, "", fixed = TRUE)[[1]]
  n <- length(chars)

  read_digits <- function(i) {
    start <- i
    while (i <= n && grepl("^[0-9]$", chars[[i]])) i <- i + 1
    list(value = paste(chars[start:(i - 1)], collapse = ""), next_i = i)
  }

  parse_parts <- function(i, stop = NULL) {
    parts <- character()
    while (i <= n && (is.null(stop) || chars[[i]] != stop)) {
      ch <- chars[[i]]
      if (ch == "(") {
        group <- parse_parts(i + 1, ")")
        i <- group$next_i + 1
        subscript <- ""
        if (i <= n && grepl("^[0-9]$", chars[[i]])) {
          digits <- read_digits(i)
          subscript <- paste0("[", digits$value, "]")
          i <- digits$next_i
        }
        parts <- c(parts, paste0("group(\"(\", ",
                                 paste(group$parts, collapse = "*"),
                                 ", \")\")", subscript))
      } else if (grepl("^[A-Z]$", ch)) {
        token <- ch
        i <- i + 1
        if (i <= n && grepl("^[a-z]$", chars[[i]])) {
          token <- paste0(token, chars[[i]])
          i <- i + 1
        }
        if (i <= n && grepl("^[0-9]$", chars[[i]])) {
          digits <- read_digits(i)
          token <- paste0(token, "[", digits$value, "]")
          i <- digits$next_i
        }
        parts <- c(parts, token)
      } else {
        parts <- c(parts, encodeString(ch, quote = "\""))
        i <- i + 1
      }
    }
    list(parts = parts, next_i = i)
  }

  paste(parse_parts(1)$parts, collapse = "*")
}

system_label_expr <- function(system_name) {
  expr_text <- vapply(strsplit(as.character(system_name), "-", fixed = TRUE),
                      function(parts) {
                        paste(vapply(parts, chemical_formula_expr, character(1)),
                              collapse = "*\"-\"*")
                      },
                      character(1))
  parse(text = expr_text)
}

pad_range <- function(values, fraction = 0.06) {
  values <- values[is.finite(values)]
  if (!length(values)) {
    return(c(0, 1))
  }
  rng <- range(values)
  span <- diff(rng)
  if (!is.finite(span) || span <= 0) {
    span <- max(abs(rng), 1)
  }
  rng + c(-1, 1) * span * fraction
}

mean_value_columns <- function(data) {
  if ("target_log_value" %in% names(data)) {
    list(
      target = "target_log_value",
      prediction = "prediction_log_value_mean",
      spread = "prediction_log_value_sd"
    )
  } else {
    list(
      target = "target_value",
      prediction = "prediction_value_mean",
      spread = "prediction_value_sd"
    )
  }
}

draw_panel <- function(df, system_name, columns) {
  target <- df[[columns$target]]
  prediction <- df[[columns$prediction]]
  spread <- df[[columns$spread]]
  keep <- is.finite(target) & is.finite(prediction)
  target <- target[keep]
  prediction <- prediction[keep]
  spread <- spread[keep]

  lim <- pad_range(c(target, prediction), 0.08)
  plot(target, prediction, type = "n", xlim = lim, ylim = lim,
       xaxt = "n", yaxt = "n",
       xlab = "", ylab = "")
  axis(1, at = pretty(lim, 3))
  axis(2, at = pretty(lim, 3))
  abline(h = pretty(lim, 4), v = pretty(lim, 4),
         col = palette_contract[["grid"]], lwd = 0.60)
  abline(0, 1, col = palette_contract[["identity"]], lwd = 1.55)

  if (length(spread) && any(is.finite(spread) & spread > 0)) {
    arrows(target, prediction - spread, target, prediction + spread,
           angle = 90, code = 3, length = 0.022, lwd = 0.75,
           col = grDevices::adjustcolor(palette_contract[["points"]], alpha.f = 0.42))
  }
  points(target, prediction, pch = 21, cex = 1.00,
         bg = grDevices::adjustcolor(palette_contract[["points_fill"]], alpha.f = 0.62),
         col = grDevices::adjustcolor("#111111", alpha.f = 0.52),
         lwd = 0.62)
  fit <- try(stats::lm(prediction ~ target), silent = TRUE)
  if (!inherits(fit, "try-error")) {
    fit_coef <- stats::coef(fit)
    if (length(fit_coef) >= 2 && all(is.finite(fit_coef[1:2]))) {
      abline(fit, col = palette_contract[["fit"]], lwd = 1.85)
    }
  }

  stats_labels <- as.expression(c(
    substitute(italic(r) == VALUE, list(VALUE = fmt(scatter_r(target, prediction)))),
    substitute(RMSE == VALUE, list(VALUE = fmt(sqrt(mean((prediction - target)^2, na.rm = TRUE))))),
    substitute(MAE == VALUE, list(VALUE = fmt(mean(abs(prediction - target), na.rm = TRUE)))),
    substitute(n == VALUE, list(VALUE = length(target)))
  ))
  usr <- par("usr")
  text(usr[1] + 0.04 * diff(usr[1:2]),
       usr[4] - c(0.09, 0.19, 0.29, 0.39) * diff(usr[3:4]),
       labels = stats_labels, adj = c(0, 1), cex = 0.78,
       col = palette_contract[["text"]])
  title(main = system_label_expr(system_name), cex.main = 1.12, font.main = 2, line = 0.20,
        col.main = palette_contract[["text"]])
}

with_device <- function(filename, width_mm, height_mm, type, draw_fun, dpi = 600) {
  width <- width_mm / 25.4
  height <- height_mm / 25.4
  if (type == "svg") {
    grDevices::svg(filename, width = width, height = height,
                   family = "Arial", pointsize = 12, bg = "white")
  } else if (type == "png") {
    grDevices::png(filename, width = width, height = height,
                   units = "in", res = dpi, type = "cairo",
                   family = "Arial", pointsize = 12, bg = "white")
  } else {
    stop("Unsupported output type: ", type)
  }
  on.exit(grDevices::dev.off(), add = TRUE)
  op <- par(no.readonly = TRUE)
  on.exit(par(op), add = TRUE)
  draw_fun()
}

save_pages <- function(base, width_mm, page_count, page_height_fun, draw_page_fun) {
  out_dirs <- c(svg = file.path(dirname(base), "fig_svg"),
                png = file.path(dirname(base), "fig_png"))
  dir.create(out_dirs[["svg"]], recursive = TRUE, showWarnings = FALSE)
  dir.create(out_dirs[["png"]], recursive = TRUE, showWarnings = FALSE)
  stem <- basename(base)
  old_pages <- c(
    list.files(out_dirs[["svg"]], paste0("^", stem, "_(a4_)?page_[0-9]+\\.svg$"),
               full.names = TRUE),
    list.files(out_dirs[["png"]], paste0("^", stem, "_(a4_)?page_[0-9]+\\.png$"),
               full.names = TRUE)
  )
  if (length(old_pages)) {
    unlink(old_pages)
  }
  written <- character()

  for (page_i in seq_len(page_count)) {
    page_stem <- paste0(stem, "_page_", sprintf("%02d", page_i))
    height_mm <- page_height_fun(page_i)
    for (ext in c("svg", "png")) {
      out_path <- file.path(out_dirs[[ext]], paste0(page_stem, ".", ext))
      with_device(out_path, width_mm, height_mm, ext,
                  function() draw_page_fun(page_i), dpi = 600)
      written <- c(written, out_path)
    }
  }

  written
}

draw_shared_x_label <- function(label_expr, line = 0.7, cex = 1.00) {
  mtext(label_expr, side = 1, outer = TRUE,
        line = line, cex = cex, font = 2)
}

draw_shared_y_label <- function(label_expr, line = 0.7, cex = 1.00) {
  mtext(label_expr, side = 2, outer = TRUE,
        line = line, cex = cex, font = 2, las = 0)
}

draw_shared_title <- function(label, y, cex = 1.34) {
  op <- par(no.readonly = TRUE)
  on.exit(par(op), add = TRUE)
  par(fig = c(0, 1, 0, 1), mar = c(0, 0, 0, 0),
      oma = c(0, 0, 0, 0), new = TRUE, xpd = NA)
  plot.new()
  text(0.012, y, labels = label, adj = c(0, 1),
       font = 2, cex = cex, col = palette_contract[["text"]])
}

render_property <- function(property_slug, model_id) {
  property_info <- property_defaults[property_defaults$slug == property_slug, ]
  if (!nrow(property_info)) {
    stop("Unknown property: ", property_slug)
  }
  if (!(model_id %in% names(model_labels))) {
    stop("Unknown model: ", model_id)
  }

  processed_root <- file.path(experiment_dir, property_slug, processed_dir_name)
  input_path <- file.path(processed_root,
                          paste0(property_slug, "_test_predictions_by_n_mean.csv"))
  if (!file.exists(input_path)) {
    stop("Missing test prediction file: ", input_path)
  }

  data <- read.csv(input_path, stringsAsFactors = FALSE)
  available_models <- sort(unique(as.character(data$N)))
  if (!(model_id %in% available_models)) {
    stop("Requested model '", model_id, "' not found for ", property_slug,
         ". Available models: ", paste(available_models, collapse = ", "))
  }

  columns <- mean_value_columns(data)
  missing_columns <- setdiff(unlist(columns), names(data))
  if (length(missing_columns)) {
    stop("Missing prediction columns in ", input_path, ": ",
         paste(missing_columns, collapse = ", "))
  }

  plot_data <- data[as.character(data$N) == model_id, ]
  systems <- sort(unique(plot_data$system))
  if (!length(systems)) {
    stop("No systems to plot for ", property_slug, " model ", model_id)
  }

  n_cols <- 4
  max_rows_per_page <- 6
  systems_per_page <- n_cols * max_rows_per_page
  page_count <- ceiling(length(systems) / systems_per_page)
  width_mm <- 210
  max_height_mm <- 280
  full_outer_margin_height_mm <- 46
  compact_outer_margin_height_mm <- 20
  panel_row_height_mm <- (max_height_mm - full_outer_margin_height_mm) / max_rows_per_page
  full_oma <- c(3.6, 3.6, 5.2, 0.7)
  compact_oma <- c(3.2, 3.6, 2.2, 0.6)

  base <- file.path(
    processed_root,
    paste0(property_slug, "_test_system_correlation_grid_", safe_model_name(model_id))
  )

  draw_page <- function(page_i) {
    page_start <- (page_i - 1) * systems_per_page + 1
    page_end <- min(page_i * systems_per_page, length(systems))
    page_systems <- systems[page_start:page_end]
    page_rows <- ceiling(length(page_systems) / n_cols)
    use_full_page_layout <- page_rows >= 4
    oma_values <- if (use_full_page_layout) full_oma else compact_oma

    par(family = "Arial", bty = "l", las = 1, tcl = -0.18,
        lwd = 0.95, lend = "round", cex.axis = 0.88,
        oma = oma_values,
        mfrow = c(page_rows, n_cols))

    for (i in seq_len(page_rows * n_cols)) {
      if (i <= length(page_systems)) {
        par(mar = c(1.78, 2.00, 1.58, 0.50),
            mgp = c(1.48, 0.44, 0))
        system_name <- page_systems[[i]]
        draw_panel(plot_data[plot_data$system == system_name, ],
                   system_name, columns)
      } else {
        par(mar = c(0, 0, 0, 0))
        plot.new()
      }
    }

    draw_shared_title(sprintf("%s test systems | %s",
                              property_info$title[[1]], display_model_name(model_id)),
                      y = if (use_full_page_layout) 0.985 else 0.975,
                      cex = 1.34)
    draw_shared_x_label(target_axis_label(property_slug),
                        line = if (use_full_page_layout) 0.7 else 1.0,
                        cex = 1.00)
    draw_shared_y_label(prediction_axis_label(property_slug),
                        line = 0.7,
                        cex = 1.00)
  }

  page_height <- function(page_i) {
    page_start <- (page_i - 1) * systems_per_page + 1
    page_end <- min(page_i * systems_per_page, length(systems))
    page_rows <- ceiling(length(systems[page_start:page_end]) / n_cols)
    if (page_rows >= 4) {
      max_height_mm
    } else {
      compact_outer_margin_height_mm + panel_row_height_mm * page_rows
    }
  }

  written_files <- save_pages(base, width_mm, page_count, page_height, draw_page)
  message("Wrote ", length(written_files), " files for ", property_slug, " / ",
          display_model_name(model_id), ":")
  for (path in written_files) {
    message("  ", path)
  }
  message("Layout: ", n_cols, " columns x up to ", max_rows_per_page,
          " rows per page; ", page_count, " pages; width ", width_mm,
          " mm; page heights ", paste(vapply(seq_len(page_count), page_height, numeric(1)),
                                      collapse = ", "),
          " mm at 600 dpi")
  invisible(written_files)
}

configs <- requested_properties(args)
all_written <- character()
for (i in seq_len(nrow(configs))) {
  all_written <- c(all_written, render_property(configs$slug[[i]], configs$model[[i]]))
}
message("Total files written: ", length(all_written))
