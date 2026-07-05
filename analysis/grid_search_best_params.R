
source("analysis/stree-code.R")
library(svmodt)
library(dplyr)
library(purrr)
library(rsample)
library(caret)

# <U+2500><U+2500> Datasets <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
ctg3 <- read.table("data/cardiotocography-3clases_R.dat") |>
  mutate(clase = as.factor(clase)) |>
  standard_scaler()

ctg10 <- read.table("data/cardiotocography-10clases_R.dat") |>
  mutate(clase = as.factor(clase)) |>
  standard_scaler()

australian_credit <- read.table("data/statlog-australian-credit_R.dat") |>
  mutate(clase = as.factor(clase)) |>
  standard_scaler()

wdbc <- read.table("data/breast-cancer-wisc-diag_R.dat") |>
  mutate(clase = as.factor(clase)) |>
  standard_scaler()

iris_data <- read.table("data/iris_R.dat") |>
  mutate(clase = as.factor(clase)) |>
  standard_scaler()

echocardiogram <- read.table("data/echocardiogram_R.dat") |>
  mutate(clase = as.factor(clase)) |>
  standard_scaler()

fertility <- read.table("data/fertility_R.dat") |>
  mutate(clase = as.factor(clase)) |>
  standard_scaler()

wine <- read.table("data/wine_R.dat") |>
  mutate(clase = as.factor(clase)) |>
  standard_scaler()

ionosphere <- read.table("data/ionosphere_R.dat") |>
  mutate(clase = as.factor(clase)) |>
  standard_scaler()

dermatology <- read.table("data/dermatology_R.dat") |>
  mutate(clase = as.factor(clase)) |>
  standard_scaler()

datasets <- list(
  wdbc           = wdbc,
  iris           = iris_data,
  echocardiogram = echocardiogram,
  fertility      = fertility,
  wine           = wine,
  ctg3           = ctg3,
  ctg10          = ctg10,
  ionosphere     = ionosphere,
  dermatology    = dermatology,
  aus_credit     = australian_credit
)

seed_list <- c(57, 31, 1714, 17, 23, 79, 83, 97, 7, 1)
grid_search_seeds <- seed_list[1:3]

# <U+2500><U+2500> Parameter Grid <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
# max_depth: removed
# min_samples: removed <U+2014> fixed at default in svm_split
# impurity_measure: fixed at "entropy"
param_grid_base <- expand.grid(
  max_depth = 10,
  feature_method         = c("random", "mutual", "cor"),
  class_weights          = c("none", "balanced"),
  max_features_strategy  = c("constant", "decrease", "random"),
  penalize_used_features = c(TRUE, FALSE),
  feature_penalty_weight = c(0.3, 0.6),
  min_impurity_decrease  = c(0.001, 0.01, 0.1),
  stringsAsFactors       = FALSE
)

param_grid_base <- bind_rows(
  param_grid_base |> filter(penalize_used_features == TRUE),
  param_grid_base |>
    filter(penalize_used_features == FALSE) |>
    filter(feature_penalty_weight == min(feature_penalty_weight)) |>
    distinct()
)

cat("Base parameter combinations:", nrow(param_grid_base), "\n")

# <U+2500><U+2500> Dataset-Adaptive Max Features <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
get_max_features_candidates <- function(data, response) {
  n_features <- ncol(data) - 1
  candidates <- unique(c(
    floor(sqrt(n_features)),
    n_features
  ))
  candidates[candidates >= 2]
}

# <U+2500><U+2500> Expand Grid <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
expand_param_grid <- function(param_grid_base, data, response) {
  mf_candidates <- get_max_features_candidates(data, response)
  cat("  max_features candidates:", paste(mf_candidates, collapse = ", "), "\n")

  expanded <- bind_rows(
    param_grid_base |>
      filter(max_features_strategy %in% c("constant", "decrease")) |>
      cross_join(data.frame(max_features = mf_candidates)),
    param_grid_base |>
      filter(max_features_strategy == "random") |>
      mutate(max_features = NA_integer_)
  )

  cat("  Expanded combinations :", nrow(expanded), "\n")
  return(expanded)
}

# <U+2500><U+2500> Build Args <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
# max_depth removed
# min_samples removed <U+2014> not a tuned parameter
# impurity_measure fixed at "entropy"
build_args <- function(params, data, response) {
  penalize <- isTRUE(params$penalize_used_features)
  strategy <- as.character(params$max_features_strategy)
  method <- as.character(params$feature_method)
  n_features <- ncol(data) - 1

  args <- list(
    data = data,
    response = response,
    feature_method = method,
    max_depth = 10L,
    class_weights = as.character(params$class_weights),
    max_features_strategy = strategy,
    penalize_used_features = penalize,
    impurity_measure = "entropy", # fixed
    min_impurity_decrease = as.numeric(params$min_impurity_decrease),
    verbose = FALSE
  )

  if (penalize) {
    args$feature_penalty_weight <- as.numeric(params$feature_penalty_weight)
  }

  if (strategy == "random") {
    args$max_features_random_range <- c(0.1, 0.9)
  } else if (strategy == "decrease") {
    mf <- if (!is.null(params$max_features) && !is.na(params$max_features)) {
      as.integer(params$max_features)
    } else {
      floor(sqrt(n_features))
    }
    args$max_features <- mf
    args$max_features_decrease_rate <- 0.5
  } else {
    mf <- if (!is.null(params$max_features) && !is.na(params$max_features)) {
      as.integer(params$max_features)
    } else {
      floor(sqrt(n_features))
    }
    args$max_features <- mf
  }

  if (method == "random") args$n_subsets <- 10L

  # Final guard
  if (!is.null(args$max_features) && is.na(args$max_features)) {
    args$max_features <- floor(sqrt(n_features))
  }

  return(args)
}

# <U+2500><U+2500> Custom caret Model <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
# max_depth, min_samples, and impurity_measure removed from parameters
svmodt_caret_model <- list(
  label = "SVMODT",
  library = "svmodt",
  type = "Classification",
  loop = NULL,
  tags = "Classification",
  parameters = data.frame(
    parameter = c(
      "max_depth",
      "feature_method", "class_weights",
      "max_features_strategy", "penalize_used_features",
      "feature_penalty_weight", "max_features",
      "min_impurity_decrease"
    ),
    class = c(
      "numeric",
      "character", "character",
      "character", "logical",
      "numeric", "numeric",
      "numeric"
    ),
    label = c(
      "Max Depth",
      "Feature Method", "Class Weights",
      "Max Features Strategy", "Penalize Features",
      "Penalty Weight", "Max Features",
      "Min Impurity Decrease"
    ),
    stringsAsFactors = FALSE
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    expand_param_grid(param_grid_base, as.data.frame(x), "clase")
  },
  fit = function(x, y, wts, param, lev, last, weights, classProbs, ...) {
    train_data <- as.data.frame(x)
    train_data$clase <- y
    args <- build_args(param, train_data, "clase")
    tryCatch(
      do.call(svmodt::svm_split, args),
      error = function(e) {
        message("  [WARN] fit failed: ", conditionMessage(e))
        NULL
      }
    )
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    tryCatch(
      as.character(predict(modelFit, as.data.frame(newdata))),
      error = function(e) {
        message("  [WARN] predict failed: ", conditionMessage(e))
        rep(NA_character_, nrow(newdata))
      }
    )
  },
  prob = NULL,
  sort = function(x) x[order(x$feature_method), ]
)

# <U+2500><U+2500> Multi-seed Grid Search <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
run_multiseed_grid_search <- function(data, response, param_grid_base,
                                      seeds = grid_search_seeds,
                                      n_folds = 5) {
  tuning_grid <- expand_param_grid(param_grid_base, data, response)
  cat("  Tuning grid rows:", nrow(tuning_grid), "\n")

  X <- data |> select(-all_of(response))
  y <- data[[response]]

  seed_results <- map(seeds, function(s) {
    cat("  Running CV with seed:", s, "\n")
    set.seed(s)

    ctrl <- trainControl(
      method          = "cv",
      number          = n_folds,
      classProbs      = FALSE,
      savePredictions = "final",
      verboseIter     = FALSE,
      allowParallel   = FALSE
    )

    tryCatch(
      {
        fit <- train(
          x         = X,
          y         = y,
          method    = svmodt_caret_model,
          trControl = ctrl,
          tuneGrid  = tuning_grid,
          metric    = "Accuracy"
        )
        fit$results |> select(-any_of(c("AccuracySD", "AccuracyMean")))
      },
      error = function(e) {
        message("  [WARN] seed ", s, " failed: ", conditionMessage(e))
        NULL
      }
    )
  })

  seed_results <- Filter(Negate(is.null), seed_results)

  if (length(seed_results) == 0) {
    message("  [ERROR] all seeds failed for this dataset")
    return(NULL)
  }

  param_cols <- names(tuning_grid)

  combined <- seed_results |>
    bind_rows(.id = "seed_idx") |>
    group_by(across(all_of(param_cols))) |>
    summarise(
      mean_acc = mean(Accuracy, na.rm = TRUE),
      sd_acc   = sd(Accuracy, na.rm = TRUE),
      n_seeds  = n(),
      .groups  = "drop"
    ) |>
    arrange(desc(mean_acc), sd_acc)

  return(combined)
}

# <U+2500><U+2500> Run Grid Search Across All Datasets <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
grid_search_results <- list()
best_params_list <- list()

for (dataset_name in names(datasets)) {
  cat("Grid search:", dataset_name, "\n")

  grid_search_results[[dataset_name]] <- run_multiseed_grid_search(
    data            = datasets[[dataset_name]],
    response        = "clase",
    param_grid_base = param_grid_base,
    seeds           = grid_search_seeds,
    n_folds         = 5
  )

  if (!is.null(grid_search_results[[dataset_name]])) {
    best <- grid_search_results[[dataset_name]] |> slice(1)

    cat("  Best mean acc          :", round(best$mean_acc, 4), "\n")
    cat("  sd across seeds        :", round(best$sd_acc, 4), "\n")
    cat("  max_features           :", ifelse(is.na(best$max_features),
      "random range [0.1, 0.9]",
      best$max_features
    ), "\n")
    cat("  mf_strategy            :", best$max_features_strategy, "\n")
    cat("  feature_method         :", best$feature_method, "\n")
    cat("  class_weights          :", best$class_weights, "\n")
    cat("  min_impurity_decrease  :", best$min_impurity_decrease, "\n")
    cat("  penalize_used_features :", best$penalize_used_features, "\n")
    cat("  feature_penalty_weight :", ifelse(isTRUE(best$penalize_used_features),
      best$feature_penalty_weight,
      "N/A"
    ), "\n")

    best_params_list[[dataset_name]] <- best |>
      mutate(dataset = dataset_name) |>
      select(dataset, everything())
  }
}

saveRDS(grid_search_results, "analysis/results/grid_search_results.rds")

best_params <- bind_rows(best_params_list)
saveRDS(best_params, "analysis/results/grid_search_best_params.rds")
print(best_params)

# <U+2500><U+2500> Rerun 10 <U+00D7> 5-fold CV with best params <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>

best_params <- readRDS("analysis/results/grid_search_best_params.rds")

r_svmodt_star_results <- list()

dataset_max_features <- map(datasets, function(dat) {
  get_max_features_candidates(dat, "clase")
})

for (dataset_name in names(datasets)) {
  bp <- best_params |>
    filter(dataset == dataset_name) |>
    slice(1)

  cat("\nProcessing:", dataset_name, "\n")

  bp_chr <- function(col) as.character(bp[[col]])
  bp_num <- function(col) as.numeric(bp[[col]])
  bp_lgl <- function(col) isTRUE(as.logical(bp[[col]]))
  has_col <- function(col) col %in% colnames(bp) && !is.na(bp[[col]])

  penalize <- if (has_col("penalize_used_features")) bp_lgl("penalize_used_features") else FALSE
  strategy <- if (has_col("max_features_strategy")) bp_chr("max_features_strategy") else "constant"
  method <- if (has_col("feature_method")) bp_chr("feature_method") else "mutual"
  n_features <- ncol(datasets[[dataset_name]]) - 1

  best_args <- list(
    feature_method         = method,
    class_weights          = if (has_col("class_weights")) bp_chr("class_weights") else "none",
    max_features_strategy  = strategy,
    penalize_used_features = penalize,
    impurity_measure       = "entropy", # fixed
    min_impurity_decrease  = if (has_col("min_impurity_decrease")) bp_num("min_impurity_decrease") else 0.0,
    verbose                = FALSE
  )

  if (penalize && has_col("feature_penalty_weight")) {
    best_args$feature_penalty_weight <- bp_num("feature_penalty_weight")
  }

  if (strategy == "random") {
    best_args$max_features_random_range <- c(0.1, 0.9)
  } else if (strategy == "decrease") {
    best_args$max_features <- if (has_col("max_features")) as.integer(bp_num("max_features")) else floor(sqrt(n_features))
    best_args$max_features_decrease_rate <- 0.5
  } else {
    best_args$max_features <- if (has_col("max_features")) as.integer(bp_num("max_features")) else floor(sqrt(n_features))
  }

  if (method == "random") best_args$n_subsets <- 10L

  cat("  strategy             :", strategy, "\n")
  cat("  method               :", method, "\n")
  cat("  min_impurity_decrease:", best_args$min_impurity_decrease, "\n")
  cat("  penalize             :", penalize, "\n")
  cat("  max_features         :", ifelse(strategy == "random",
    "range [0.1, 0.9]",
    best_args$max_features
  ), "\n")

  fold_accs <- numeric(10)

  for (i in 1:10) {
    set.seed(seed_list[i])
    folds <- vfold_cv(datasets[[dataset_name]], v = 5, strata = clase)

    fold_accs[i] <- mean(map_dbl(folds$splits, function(split) {
      train_data <- analysis(split)
      test_data <- assessment(split)
      tryCatch(
        {
          model <- do.call(
            svmodt::svm_split,
            c(list(data = train_data, response = "clase", max_depth = 10), best_args)
          )
          preds <- predict(model, test_data)
          mean(preds == test_data$clase, na.rm = TRUE)
        },
        error = function(e) {
          message(
            "  [WARN] ", dataset_name, " seed ", seed_list[i],
            ": ", conditionMessage(e)
          )
          NA_real_
        }
      )
    }), na.rm = TRUE)

    cat("  Seed", seed_list[i], "| Acc:", round(fold_accs[i], 4), "\n")
  }

  r_svmodt_star_results[[dataset_name]] <- fold_accs
  cat("  Overall mean         :", round(mean(fold_accs, na.rm = TRUE), 4), "\n")
}

r_svmodt_star <- do.call(cbind, r_svmodt_star_results)
colnames(r_svmodt_star) <- names(datasets)
saveRDS(r_svmodt_star, "analysis/results/r_svmodt_star.rds")
