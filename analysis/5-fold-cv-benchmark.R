# Libraries
library(rsample)
library(dplyr)
library(purrr)
library(reticulate)
library(svmodt)
source("analysis/stree-code.R")

# install.packages("../project-svodt/", repos = NULL, type = "source")

# Libraries - Python
stree <- import("stree")
sklearn_svm <- import("sklearn.svm")

# Default STree Arguments
stree_args <- list(
  C = 1,
  tol = 0.0001,
  kernel = "linear",
  max_iter = 10000000L,
  max_depth = 10
)

#  Datasets
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

iris <- read.table("data/iris_R.dat") |>
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
  iris           = iris,
  echocardiogram = echocardiogram,
  fertility      = fertility,
  wine           = wine,
  ctg3           = ctg3,
  ctg10          = ctg10,
  ionosphere     = ionosphere,
  dermatology    = dermatology,
  aus_credit     = australian_credit
)





dataset_col_names <- paste0("stat_", names(datasets))
seed_list <- c(57, 31, 1714, 17, 23, 79, 83, 97, 7, 1)

#  Shared Prediction Functions
predict_svmodt <- function(model, newdata, ...) {
  predict(model, newdata)
}

predict_stree <- function(model, newdata, ...) {
  stree_predict(model, newdata)
}

predict_python_stree <- function(model, test_data, response) {
  X_test <- test_data |> dplyr::select(-all_of(response))
  preds <- model$predict(X_test)
  py_to_r(preds)
}

#  Base SVMODT args (fixed across ALL experiments)
base_svmodt_args <- list(
  max_depth             = 15,
  verbose               = FALSE
)

#  Training Function Factory
make_train_svmodt <- function(extra_args) {
  function(data, response) {
    do.call(
      svmodt::svm_split,
      c(
        list(data = data, response = response),
        base_svmodt_args,
        extra_args
      ) # extra_args override base where they overlap
    )
  }
}

#  Generic Experiment Runner
#  Helper: Class distribution
get_class_counts <- function(data, response) {
  counts <- table(data[[response]])
  tibble(
    class = names(counts),
    n     = as.integer(counts)
  )
}

# Print class distributions for all datasets
cat("Class distributions:\n")
walk2(names(datasets), datasets, function(name, dat) {
  cat(" ", name, ":\n")
  cc <- get_class_counts(dat, "clase")
  walk2(cc$class, cc$n, ~ cat("    class", .x, ":", .y, "observations\n"))
})

#  Helper: Multiclass F1 (macro)
compute_f1_macro <- function(preds, truth) {
  classes <- levels(as.factor(truth))
  f1s <- map_dbl(classes, function(cls) {
    tp <- sum(preds == cls & truth == cls)
    fp <- sum(preds == cls & truth != cls)
    fn <- sum(preds != cls & truth == cls)
    precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)
  })
  mean(f1s, na.rm = TRUE)
}

compute_f1_per_class <- function(preds, truth) {
  classes <- levels(as.factor(truth))
  f1s <- map_dbl(classes, function(cls) {
    tp <- sum(preds == cls & truth == cls)
    fp <- sum(preds == cls & truth != cls)
    fn <- sum(preds != cls & truth == cls)
    precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)
  })
  setNames(f1s, classes)
}

#  Updated Generic Experiment Runner
run_experiment <- function(variants, datasets, seed_list,
                           dataset_max_features = NULL,
                           compute_f1 = FALSE,
                           n_iter = 10) {
  results <- map(names(variants), ~ list()) |> setNames(names(variants))
  results_f1 <- map(names(variants), ~ list()) |> setNames(names(variants))

  for (i in 1:n_iter) {
    cat("Iteration", i, "\n")

    set.seed(seed_list[i])
    folds_list <- map(datasets, ~ vfold_cv(.x, v = 5, strata = clase))

    for (dataset_name in names(datasets)) {
      cat("  Processing", dataset_name, "\n")

      mf_candidates <- dataset_max_features[[dataset_name]]
      current_folds <- folds_list[[dataset_name]]

      results_fold <- map_df(current_folds$splits, function(split) {
        train_data <- analysis(split)
        test_data <- assessment(split)

        fold_accs <- map_dfc(names(variants), function(variant_name) {
          variant_args <- variants[[variant_name]]

          if (!is.null(mf_candidates) &&
            !isTRUE(variant_args$max_features_strategy %in% c("random"))) {
            variant_args$max_features <- sample(mf_candidates, 1)
          }

          train_fn <- make_train_svmodt(variant_args)

          tryCatch(
            {
              model <- train_fn(train_data, "clase")
              preds <- predict_svmodt(model, test_data)
              truth <- test_data$clase

              acc <- mean(preds == truth, na.rm = TRUE)

              if (compute_f1) {
                f1 <- compute_f1_macro(preds, truth)
                tibble(
                  !!paste0(variant_name, "_acc") := acc,
                  !!paste0(variant_name, "_f1") := f1
                )
              } else {
                tibble(!!variant_name := acc)
              }
            },
            error = function(e) {
              message(
                "  [WARN] ", variant_name, " - ", dataset_name,
                ": ", conditionMessage(e)
              )
              if (compute_f1) {
                tibble(
                  !!paste0(variant_name, "_acc") := NA_real_,
                  !!paste0(variant_name, "_f1") := NA_real_
                )
              } else {
                tibble(!!variant_name := NA_real_)
              }
            }
          )
        })

        fold_accs
      })

      # Store accuracy
      for (variant_name in names(variants)) {
        acc_col <- if (compute_f1) paste0(variant_name, "_acc") else variant_name
        results[[variant_name]][[dataset_name]][i] <- mean(results_fold[[acc_col]],
          na.rm = TRUE
        )
      }

      # Store F1 if requested
      if (compute_f1) {
        for (variant_name in names(variants)) {
          f1_col <- paste0(variant_name, "_f1")
          results_f1[[variant_name]][[dataset_name]][i] <- mean(results_fold[[f1_col]],
            na.rm = TRUE
          )
        }
      }
    }
  }

  out <- list(accuracy = results)
  if (compute_f1) out$f1 <- results_f1

  return(out)
}

#  Updated to_matrices
to_matrices <- function(results, col_names) {
  # Unwrap nested structure from compute_f1 = TRUE output
  if ("accuracy" %in% names(results)) results <- results$accuracy
  if ("f1" %in% names(results)) results <- results$f1

  # Validate length matches col_names before binding
  n_datasets <- length(results[[1]])
  if (n_datasets != length(col_names)) {
    stop(
      "to_matrices: col_names length (", length(col_names), ") ",
      "does not match number of datasets (", n_datasets, "). ",
      "Check dataset_col_names matches datasets list."
    )
  }

  map(results, function(res) {
    # Guard: ensure all dataset entries have the same length
    entry_lengths <- sapply(res, length)
    if (length(unique(entry_lengths)) > 1) {
      warning(
        "Unequal iteration counts across datasets: ",
        paste(names(res), "=", entry_lengths, collapse = ", "),
        " <U+2014> padding shorter entries with NA"
      )
      max_len <- max(entry_lengths)
      res <- lapply(res, function(x) {
        length(x) <- max_len
        x
      })
    }

    mat <- do.call(cbind, res)
    colnames(mat) <- col_names
    mat
  })
}


# EXPERIMENT 1: Benchmark (R STree, SVMODT default, Python STree)

cat("\n====== EXPERIMENT 1: Benchmark ======\n")

results_bench <- list(r_stree = list(), r_svmodt = list(), py_stree = list())

train_stree_default <- function(data, response) {
  stree_split(
    data             = data,
    response         = response,
    kernel           = "linear",
    impurity_measure = "entropy",
    verbose          = FALSE,
    max_depth        = 15
  )
}

train_svmodt_default <- function(data, response) {
  do.call(
    svmodt::svm_split,
    c(list(data = data, response = response), base_svmodt_args)
  )
}

train_python_stree <- function(data, response) {
  y <- data[[response]]
  X <- data |> dplyr::select(-all_of(response))
  py_model <- do.call(stree$Stree, stree_args)
  py_model$fit(X, y)
  py_model
}

for (i in 1:10) {
  cat("Iteration", i, "\n")

  set.seed(seed_list[i])
  folds_list <- map(datasets, ~ vfold_cv(.x, v = 5, strata = clase))

  for (dataset_name in names(datasets)) {
    cat("  Processing", dataset_name, "\n")

    current_folds <- folds_list[[dataset_name]]

    results_fold <- map_df(current_folds$splits, function(split) {
      train_data <- analysis(split)
      test_data <- assessment(split)

      model_r_stree <- train_stree_default(train_data, "clase")
      preds_r_stree <- predict_stree(model_r_stree, test_data)

      model_r_svmodt <- train_svmodt_default(train_data, "clase")
      preds_r_svmodt <- predict_svmodt(model_r_svmodt, test_data)

      model_py_stree <- train_python_stree(train_data, "clase")
      preds_py_stree <- predict_python_stree(model_py_stree, test_data, "clase")

      tibble(
        r_stree  = mean(preds_r_stree == test_data$clase),
        r_svmodt = mean(preds_r_svmodt == test_data$clase),
        py_stree = mean(preds_py_stree == test_data$clase)
      )
    })

    results_bench$r_stree[[dataset_name]][i] <- mean(results_fold$r_stree)
    results_bench$r_svmodt[[dataset_name]][i] <- mean(results_fold$r_svmodt)
    results_bench$py_stree[[dataset_name]][i] <- mean(results_fold$py_stree)
  }
}

r_stree <- do.call(cbind, results_bench$r_stree)
colnames(r_stree) <- dataset_col_names
r_svmodt <- do.call(cbind, results_bench$r_svmodt)
colnames(r_svmodt) <- dataset_col_names
py_stree <- do.call(cbind, results_bench$py_stree)
colnames(py_stree) <- dataset_col_names

r_stree |> saveRDS("analysis/r_stree.rds")
r_svmodt |> saveRDS("analysis/r_svmodt.rds")
py_stree |> saveRDS("analysis/py_stree.rds")


# EXPERIMENT 1b: Benchmark F1 Scores (R STree, SVMODT default, Python STree)

cat("\n====== EXPERIMENT 1b: Benchmark F1 ======\n")

results_bench_f1 <- list(
  r_stree  = list(),
  r_svmodt = list(),
  py_stree = list()
)

for (i in 1:10) {
  cat("Iteration", i, "\n")

  set.seed(seed_list[i])
  folds_list <- map(datasets, ~ vfold_cv(.x, v = 5, strata = clase))

  for (dataset_name in names(datasets)) {
    cat("  Processing", dataset_name, "\n")

    current_folds <- folds_list[[dataset_name]]

    results_fold <- map_df(current_folds$splits, function(split) {
      train_data <- analysis(split)
      test_data <- assessment(split)
      truth <- test_data$clase

      # R STree
      f1_r_stree <- tryCatch(
        {
          model <- train_stree_default(train_data, "clase")
          preds <- predict_stree(model, test_data)
          compute_f1_macro(preds, truth)
        },
        error = function(e) {
          message("  [WARN] r_stree f1 - ", dataset_name, ": ", conditionMessage(e))
          NA_real_
        }
      )

      # R SVMODT
      f1_r_svmodt <- tryCatch(
        {
          model <- train_svmodt_default(train_data, "clase")
          preds <- predict_svmodt(model, test_data)
          compute_f1_macro(preds, truth)
        },
        error = function(e) {
          message("  [WARN] r_svmodt f1 - ", dataset_name, ": ", conditionMessage(e))
          NA_real_
        }
      )

      # Python STree
      f1_py_stree <- tryCatch(
        {
          model <- train_python_stree(train_data, "clase")
          preds <- predict_python_stree(model, test_data, "clase")
          compute_f1_macro(preds, truth)
        },
        error = function(e) {
          message("  [WARN] py_stree f1 - ", dataset_name, ": ", conditionMessage(e))
          NA_real_
        }
      )

      tibble(
        r_stree  = f1_r_stree,
        r_svmodt = f1_r_svmodt,
        py_stree = f1_py_stree
      )
    })

    results_bench_f1$r_stree[[dataset_name]][i] <- mean(results_fold$r_stree, na.rm = TRUE)
    results_bench_f1$r_svmodt[[dataset_name]][i] <- mean(results_fold$r_svmodt, na.rm = TRUE)
    results_bench_f1$py_stree[[dataset_name]][i] <- mean(results_fold$py_stree, na.rm = TRUE)
  }
}

# Convert to matrices
r_stree_f1 <- do.call(cbind, results_bench_f1$r_stree)
colnames(r_stree_f1) <- dataset_col_names
r_svmodt_f1 <- do.call(cbind, results_bench_f1$r_svmodt)
colnames(r_svmodt_f1) <- dataset_col_names
py_stree_f1 <- do.call(cbind, results_bench_f1$py_stree)
colnames(py_stree_f1) <- dataset_col_names

r_stree_f1 |> saveRDS("analysis/r_stree_f1.rds")
r_svmodt_f1 |> saveRDS("analysis/r_svmodt_f1.rds")
py_stree_f1 |> saveRDS("analysis/py_stree_f1.rds")


# EXPERIMENT 2: Feature Selection Method
# max_features = 3 (constant), max_features_strategy = "constant" [FIXED]
# Varies: feature_method

cat("\n====== EXPERIMENT 2: Feature Selection Method ======\n")

feature_method_variants <- list(
  random = list(feature_method = "random", n_subsets = 10, max_features_strategy = "random", max_features_random_range = c(0.5, 0.5)),
  mutual = list(feature_method = "mutual", max_features_strategy = "random", max_features_random_range = c(0.5, 0.5)),
  cor    = list(feature_method = "cor", max_features_strategy = "random", max_features_random_range = c(0.5, 0.5))
)

results_feature <- run_experiment(feature_method_variants, datasets, seed_list)
matrices_feature <- to_matrices(results_feature, dataset_col_names)

matrices_feature$random |> saveRDS("analysis/svmodt_feature_random.rds")
matrices_feature$mutual |> saveRDS("analysis/svmodt_feature_mutual.rds")
matrices_feature$cor |> saveRDS("analysis/svmodt_feature_cor.rds")


# EXPERIMENT 3: Max Features Strategy
# feature_method = "mutual" [FIXED]
# Varies: max_features_strategy (and max_features / range accordingly)

cat("\n====== EXPERIMENT 3: Max Features Strategy ======\n")

mf_strategy_variants <- list(
  constant = list(
    feature_method        = "mutual",
    max_features_strategy = "constant"
  ),
  decrease = list(
    feature_method = "mutual",
    max_features_strategy = "decrease",
    max_features_decrease_rate = 0.5
  ),
  random = list(
    feature_method             = "mutual",
    max_features_strategy      = "random",
    max_features_random_range  = c(0.5, 0.8)
  )
)

results_mf <- run_experiment(mf_strategy_variants, datasets, seed_list)
matrices_mf <- to_matrices(results_mf, dataset_col_names)

matrices_mf$constant |> saveRDS("analysis/svmodt_mf_constant.rds")
matrices_mf$decrease |> saveRDS("analysis/svmodt_mf_decrease.rds")
matrices_mf$random |> saveRDS("analysis/svmodt_mf_random.rds")


# EXPERIMENT 4: Class Weights  (accuracy + F1 + class counts)

cat("\n====== EXPERIMENT 4: Class Weights ======\n")

# Class counts per dataset
class_counts <- map_df(names(datasets), function(dataset_name) {
  get_class_counts(datasets[[dataset_name]], "clase") |>
    mutate(dataset = dataset_name) |>
    select(dataset, class, n)
})

cat("\nClass counts:\n")
print(class_counts)
saveRDS(class_counts, "analysis/class_counts.rds")

class_weight_variants <- list(
  none               = list(feature_method = "mutual", class_weights = "none"),
  balanced           = list(feature_method = "mutual", class_weights = "balanced"),
  balanced_subsample = list(feature_method = "mutual", class_weights = "balanced_subsample")
)

results_weights <- run_experiment(
  variants             = class_weight_variants,
  datasets             = datasets,
  seed_list            = seed_list,
  dataset_max_features = dataset_max_features,
  compute_f1           = TRUE # F1 only for this experiment
)

# Accuracy matrices
matrices_weights_acc <- to_matrices(results_weights$accuracy, dataset_col_names)
matrices_weights_acc$none |> saveRDS("analysis/svmodt_weights_none_acc.rds")
matrices_weights_acc$balanced |> saveRDS("analysis/svmodt_weights_balanced_acc.rds")
matrices_weights_acc$balanced_subsample |> saveRDS("analysis/svmodt_weights_balanced_subsample_acc.rds")

# F1 matrices
matrices_weights_f1 <- to_matrices(results_weights$f1, dataset_col_names)
matrices_weights_f1$none |> saveRDS("analysis/svmodt_weights_none_f1.rds")
matrices_weights_f1$balanced |> saveRDS("analysis/svmodt_weights_balanced_f1.rds")
matrices_weights_f1$balanced_subsample |> saveRDS("analysis/svmodt_weights_balanced_subsample_f1.rds")


# EXPERIMENT 5: Feature Penalisation
# max_features = 3 (constant), feature_method = "mutual" [FIXED]
# Varies: penalize_used_features, feature_penalty_weight

cat("\n====== EXPERIMENT 5: Feature Penalisation ======\n")

penalise_variants <- list(
  no_penalty = list(feature_method = "mutual", max_features = 4, penalize_used_features = FALSE),
  penalty_low = list(
    feature_method = "mutual", max_features = 4, penalize_used_features = TRUE,
    feature_penalty_weight = 0.2
  ),
  penalty_medium = list(
    feature_method = "mutual", max_features = 4, penalize_used_features = TRUE,
    feature_penalty_weight = 0.5
  ),
  penalty_high = list(
    feature_method = "mutual", max_features = 4, penalize_used_features = TRUE,
    feature_penalty_weight = 0.8
  )
)

results_penalise <- run_experiment(penalise_variants, datasets, seed_list)
results_to_matrix <- function(results, col_names) {
  # Handle nested output from compute_f1 = TRUE
  if ("accuracy" %in% names(results)) results <- results$accuracy

  # Convert each variant directly
  lapply(results, function(variant_results) {
    mat <- matrix(
      unlist(variant_results),
      nrow = length(variant_results[[1]]),
      ncol = length(variant_results),
      dimnames = list(NULL, col_names)
    )
    mat
  })
}

# Usage - penalise
mats <- results_to_matrix(results_penalise, dataset_col_names)

svmodt_penalty_none <- mats$no_penalty
svmodt_penalty_low <- mats$penalty_low
svmodt_penalty_medium <- mats$penalty_medium
svmodt_penalty_high <- mats$penalty_high

svmodt_penalty_none |> saveRDS("analysis/svmodt_penalty_none.rds")
svmodt_penalty_low |> saveRDS("analysis/svmodt_penalty_low.rds")
svmodt_penalty_medium |> saveRDS("analysis/svmodt_penalty_medium.rds")
svmodt_penalty_high |> saveRDS("analysis/svmodt_penalty_high.rds")
