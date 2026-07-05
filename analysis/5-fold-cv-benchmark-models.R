library(rsample)
library(dplyr)
library(purrr)
library(reticulate)
library(svmodt)
library(e1071)
library(rpart)
library(aorsf)
library(nnet)
library(PPtreeExt)
library(kableExtra)
source("analysis/stree-code.R")

# <U+2500><U+2500> Python <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
stree <- import("stree")
sklearn_svm <- import("sklearn.svm")

# <U+2500><U+2500> Datasets <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
ctg3 <- read.table("data/cardiotocography-3clases_R.dat")
ctg3[, -ncol(ctg3)] <- lapply(ctg3[, -ncol(ctg3)], standard_scaler)

ctg10 <- read.table("data/cardiotocography-10clases_R.dat")
ctg10[, -ncol(ctg10)] <- lapply(ctg10[, -ncol(ctg10)], standard_scaler)

australian_credit <- read.table("data/statlog-australian-credit_R.dat")
australian_credit[, -ncol(australian_credit)] <- lapply(australian_credit[, -ncol(australian_credit)], standard_scaler)

wdbc <- read.table("data/breast-cancer-wisc-diag_R.dat")
wdbc[, -ncol(wdbc)] <- lapply(wdbc[, -ncol(wdbc)], standard_scaler)

iris <- read.table("data/iris_R.dat")
iris[, -ncol(iris)] <- lapply(iris[, -ncol(iris)], standard_scaler)

echocardiogram <- read.table("data/echocardiogram_R.dat")
echocardiogram[, -ncol(echocardiogram)] <- lapply(echocardiogram[, -ncol(echocardiogram)], standard_scaler)

fertility <- read.table("data/fertility_R.dat")
fertility[, -ncol(fertility)] <- lapply(fertility[, -ncol(fertility)], standard_scaler)

wine <- read.table("data/wine_R.dat")
wine[, -ncol(wine)] <- lapply(wine[, -ncol(wine)], standard_scaler)

ionosphere <- read.table("data/ionosphere_R.dat")
ionosphere[, -ncol(ionosphere)] <- lapply(ionosphere[, -ncol(ionosphere)], standard_scaler)

dermatology <- read.table("data/dermatology_R.dat")
dermatology[, -ncol(dermatology)] <- lapply(dermatology[, -ncol(dermatology)], standard_scaler)

datasets <- list(
  wdbc           = wdbc,
  iris           = iris,
  echocardiogram = echocardiogram,
  fertility      = fertility,
  wine           = wine,
  ctg3           = ctg3,
  ctg10          = ctg10,
  ionosphere     = ionosphere,
  dermatology    = dermatology
)

dataset_col_names <- paste0("stat_", names(datasets))
seed_list <- c(57, 31, 1714, 17, 23, 79, 83, 97, 7, 1)

data_names <- c(
  "WDBC Diagnosis", "Iris", "Echocardiogram", "Fertility", "Wine",
  "Cardiotography-3", "Cardiotography-10", "Ionosphere",
  "Dermatology"
)
data_num_observations <- c(569, 150, 131, 100, 178, 2126, 2126, 351, 366)
data_num_features <- c(30, 4, 10, 9, 12, 21, 21, 33, 34)
data_num_classes <- c(2, 3, 2, 2, 3, 3, 10, 2, 6)

# <U+2500><U+2500> Python STree best configurations per dataset <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
stree_best_args <- list(
  wdbc = list(C = 0.2, max_iter = 1e4L, kernel = "linear"),
  iris = list(max_iter = 10000000L, kernel = "linear"),
  echocardiogram = list(
    C = 7, gamma = 0.1, kernel = "poly",
    max_features = "auto", max_iter = 1e4L
  ),
  fertility = list(
    C = 0.05, max_features = "auto", max_iter = 1e4L,
    kernel = "linear"
  ),
  wine = list(C = 0.55, max_iter = 1e4L, kernel = "linear"),
  ctg3 = list(max_iter = 10000000L, kernel = "linear"),
  ctg10 = list(max_iter = 10000000L, kernel = "linear"),
  ionosphere = list(
    C = 7, gamma = 0.1, kernel = "rbf",
    max_iter = 1e4L
  ),
  dermatology = list(C = 55, max_iter = 1e4L, kernel = "linear"),
  aus_credit = list(
    C = 0.05, max_features = "auto", max_iter = 1e4L,
    kernel = "linear"
  )
)

# <U+2500><U+2500> Training Functions <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>

train_rpart_shallow <- function(data, response) {
  data[[response]] <- as.factor(data[[response]])
  f <- as.formula(paste(response, "~ ."))
  rpart::rpart(f,
    data = data,
    control = rpart::rpart.control(maxdepth = 3, cp = 0.01)
  )
}

train_rpart_deep <- function(data, response) {
  data[[response]] <- as.factor(data[[response]])
  f <- as.formula(paste(response, "~ ."))
  rpart::rpart(f,
    data = data,
    control = rpart::rpart.control(maxdepth = 15, cp = 0.001)
  )
}

train_svm_linear <- function(data, response) {
  data[[response]] <- as.factor(data[[response]])
  X <- data |> select(-all_of(response))
  y <- data[[response]]
  e1071::svm(
    x = X, y = y, kernel = "linear",
    scale = FALSE, probability = TRUE
  )
}

train_svm_rbf <- function(data, response) {
  data[[response]] <- as.factor(data[[response]])
  X <- data |> select(-all_of(response))
  y <- data[[response]]
  e1071::svm(
    x = X, y = y, kernel = "radial",
    scale = FALSE, probability = TRUE
  )
}

train_aorsf <- function(data, response) {
  data[[response]] <- as.factor(data[[response]])
  f <- as.formula(paste(response, "~ ."))
  aorsf::orsf(f, data = data, control = orsf_control_classification(), verbose_progress = FALSE)
}

train_logreg <- function(data, response) {
  data[[response]] <- as.factor(data[[response]])
  lvls <- levels(data[[response]])
  n_classes <- length(lvls)
  f <- as.formula(paste(response, "~ ."))

  if (n_classes == 2) {
    data_glm <- data
    data_glm[[response]] <- as.integer(data[[response]]) - 1L
    model <- glm(f, data = data_glm, family = binomial())
    # Store training levels for safe prediction
    model$.response_levels <- lvls
    model
  } else {
    nnet::multinom(f, data = data, trace = FALSE, MaxNWts = 10000)
  }
}

train_pptree <- function(data, response) {
  data[[response]] <- as.factor(data[[response]])
  PPtreeExt::PPtreeExtclass(
    as.formula(paste(response, "~ .")),
    data = data
  )
}

train_python_stree_best <- function(data, response, dataset_name) {
  data[[response]] <- as.factor(data[[response]])
  y <- data[[response]]
  X <- data |> dplyr::select(-all_of(response))
  args <- c(list(tol = 0.0001), stree_best_args[[dataset_name]])
  py_model <- do.call(stree$Stree, args)
  py_model$fit(X, y)
  py_model
}

# <U+2500><U+2500> rpart Grid <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
rpart_param_grid <- expand.grid(
  maxdepth = c(2, 3, 5, 8, 15, 30),
  cp = c(0.1, 0.01, 0.001, 0.0001),
  stringsAsFactors = FALSE
)

train_rpart_grid <- function(data, response, k_inner = 3) {
  data[[response]] <- as.factor(data[[response]])
  f <- as.formula(paste(response, "~ ."))

  # Inner CV to pick best hyperparams
  inner_folds <- rsample::vfold_cv(data, v = k_inner, strata = all_of(response))

  best_acc <- -Inf
  best_params <- rpart_param_grid[1, ]

  for (i in seq_len(nrow(rpart_param_grid))) {
    params <- rpart_param_grid[i, ]

    fold_accs <- purrr::map_dbl(inner_folds$splits, function(split) {
      tr <- rsample::analysis(split)
      va <- rsample::assessment(split)
      tr[[response]] <- as.factor(tr[[response]])

      m <- tryCatch(
        rpart::rpart(f,
          data = tr,
          control = rpart::rpart.control(
            maxdepth = params$maxdepth,
            cp       = params$cp
          )
        ),
        error = function(e) NULL
      )
      if (is.null(m)) {
        return(NA_real_)
      }

      preds <- as.character(predict(m, va, type = "class"))
      mean(preds == as.character(va[[response]]), na.rm = TRUE)
    })

    mean_acc <- mean(fold_accs, na.rm = TRUE)
    if (!is.na(mean_acc) && mean_acc > best_acc) {
      best_acc <- mean_acc
      best_params <- params
    }
  }

  # Refit on full training data with winning params
  rpart::rpart(f,
    data = data,
    control = rpart::rpart.control(
      maxdepth = best_params$maxdepth,
      cp       = best_params$cp
    )
  )
}

# <U+2500><U+2500> Prediction Functions <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>

predict_rpart <- function(model, newdata, response) {
  as.character(predict(model, newdata, type = "class"))
}

predict_svm_e1071 <- function(model, newdata, response) {
  X <- newdata |> select(-all_of(response))
  as.character(predict(model, X))
}

predict_aorsf <- function(model, newdata, response) {
  X <- newdata |> select(-all_of(response))
  preds <- predict(model, new_data = X, pred_type = "class")
  as.numeric(preds) - 1L
}

predict_logreg <- function(model, newdata, response) {
  if (inherits(model, "glm")) {
    lvls <- if (!is.null(model$.response_levels)) {
      model$.response_levels
    } else {
      sort(unique(as.character(newdata[[response]])))
    }
    probs <- tryCatch(
      predict(model, newdata, type = "response"),
      error = function(e) {
        message("    [WARN] glm predict: ", conditionMessage(e))
        rep(0.5, nrow(newdata))
      }
    )
    return(ifelse(probs > 0.5, lvls[2], lvls[1]))
  }

  if (inherits(model, "multinom")) {
    return(tryCatch(
      as.character(predict(model, newdata, type = "class")),
      error = function(e) {
        message("    [WARN] multinom predict: ", conditionMessage(e))
        rep(NA_character_, nrow(newdata))
      }
    ))
  }

  message(
    "    [WARN] predict_logreg: unknown model class: ",
    paste(class(model), collapse = ", ")
  )
  rep(NA_character_, nrow(newdata))
}

predict_pptree <- function(model, newdata, response) {
  true_class <- newdata[[response]]
  features <- newdata[, names(newdata) != response]
  preds <- PPtreeExt:::predict.PPtreeExtclass(object = model, newdata = features, true.class = true_class)
  preds$predict.class
}

predict_python_stree_best <- function(model, test_data, response) {
  X_test <- test_data |> dplyr::select(-all_of(response))
  preds <- model$predict(X_test)
  as.character(py_to_r(preds))
}

# <U+2500><U+2500> Benchmark Runner <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
run_benchmark <- function(datasets, seed_list, n_iter = 10) {
  model_names <- c(
    "rpart_grid",
    "rpart_shallow", "rpart_deep",
    "svm_linear",    "svm_rbf",
    "aorsf",         "logreg",
    "pptree",        "py_stree_best"
  )

  results <- map(model_names, ~ list()) |> setNames(model_names)

  for (i in 1:n_iter) {
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

        # Generic safe accuracy wrapper with model name for diagnostics
        safe_acc <- function(train_fn, pred_fn, model_name, ...) {
          tryCatch(
            {
              model <- train_fn(train_data, "clase", ...)
              preds <- pred_fn(model, test_data, "clase")
              mean(preds == as.numeric(truth), na.rm = TRUE)
            },
            error = function(e) {
              message(
                "    [WARN] ", model_name, " | ", dataset_name,
                ": ", conditionMessage(e)
              )
              NA_real_
            }
          )
        }

        tibble(
          rpart_grid = safe_acc(train_rpart_grid, predict_rpart, "rpart_grid"),
          rpart_shallow = safe_acc(train_rpart_shallow, predict_rpart, "rpart_shallow"),
          rpart_deep = safe_acc(train_rpart_deep, predict_rpart, "rpart_deep"),
          svm_linear = safe_acc(train_svm_linear, predict_svm_e1071, "svm_linear"),
          svm_rbf = safe_acc(train_svm_rbf, predict_svm_e1071, "svm_rbf"),
          aorsf = safe_acc(train_aorsf, predict_aorsf, "aorsf"),
          logreg = safe_acc(train_logreg, predict_logreg, "logreg"),
          pptree = safe_acc(train_pptree, predict_pptree, "pptree"),
          py_stree_best = tryCatch(
            {
              model <- train_python_stree_best(train_data, "clase", dataset_name)
              preds <- predict_python_stree_best(model, test_data, "clase")
              mean(preds == as.factor(as.numeric(truth)), na.rm = TRUE)
            },
            error = function(e) {
              message(
                "    [WARN] py_stree_best | ", dataset_name,
                ": ", conditionMessage(e)
              )
              NA_real_
            }
          )
        )
      })

      for (model_name in model_names) {
        results[[model_name]][[dataset_name]][i] <- mean(
          results_fold[[model_name]],
          na.rm = TRUE
        )
      }
    }
  }

  return(results)
}

# <U+2500><U+2500> Run <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
cat("\n====== BENCHMARK: Multiple Classifiers ======\n")
benchmark_results <- run_benchmark(datasets, seed_list, n_iter = 10)

# <U+2500><U+2500> Convert to Matrices <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
benchmark_matrices <- map(benchmark_results, function(res) {
  mat <- do.call(cbind, res)
  colnames(mat) <- dataset_col_names
  mat
})

# <U+2500><U+2500> Save <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
iwalk(benchmark_matrices, function(mat, name) {
  saveRDS(mat, paste0("analysis/results/benchmark_", name, ".rds"))
})

cat("\nSaved", length(benchmark_matrices), "result matrices.\n")

# <U+2500><U+2500> Summary Table <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
# <U+2500><U+2500> Load all results <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
r_svmodt_star <- readRDS("analysis/results/r_svmodt_star.rds")[, 1:9]
r_svmodt <- readRDS("analysis/results/r_svmodt.rds")[, 1:9]
logreg <- readRDS("analysis/results/benchmark_logreg.rds")[, 1:9]
rpart_shallow <- readRDS("analysis/results/benchmark_rpart_shallow.rds")[, 1:9]
rpart_grid <- readRDS("analysis/results/benchmark_rpart_grid.rds")
aorsf <- readRDS("analysis/results/benchmark_aorsf.rds")
pptree <- readRDS("analysis/results/benchmark_pptree.rds")[, 1:9]
rpart_deep <- readRDS("analysis/results/benchmark_rpart_deep.rds")[, 1:9]
svm_linear <- readRDS("analysis/results/benchmark_svm_linear.rds")[, 1:9]
py_stree_best <- readRDS("analysis/results/benchmark_py_stree_best.rds")[, 1:9]

# <U+2500><U+2500> Summary Table <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
mean_sd <- function(x, digits = 3) {
  x <- as.numeric(x)
  sprintf(
    paste0("%.", digits, "f \u00B1 %.", digits, "f"),
    mean(x, na.rm = TRUE), sd(x, na.rm = TRUE)
  )
}

tbl_benchmark <- data.frame(
  Dataset = data_names,
  N = data_num_observations,
  X = data_num_features,
  L = data_num_classes,
  r_part_grid = apply(rpart_grid, 2, mean_sd),
  rpart_shallow = apply(rpart_shallow, 2, mean_sd),
  rpart_deep = apply(rpart_deep, 2, mean_sd),
  svm_linear = apply(svm_linear, 2, mean_sd),
  aorsf = apply(aorsf, 2, mean_sd),
  logreg = apply(logreg, 2, mean_sd),
  pptree = apply(pptree, 2, mean_sd),
  py_stree_best = apply(py_stree_best, 2, mean_sd),
  svmodt_default = apply(as.matrix(r_svmodt), 2, mean_sd),
  svmodt_star = apply(as.matrix(r_svmodt_star), 2, mean_sd),
  stringsAsFactors = FALSE
)

# <U+2500><U+2500> Bold best per row <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
model_cols <- 5:ncol(tbl_benchmark)

tbl_benchmark_bold <- tbl_benchmark
tbl_benchmark_bold[model_cols] <- t(apply(tbl_benchmark[model_cols], 1, function(row) {
  means <- as.numeric(sub(" \u00B1.*", "", row))
  max_idx <- which(means == max(means, na.rm = TRUE))
  out <- row
  out[max_idx] <- cell_spec(out[max_idx], bold = TRUE)
  out
}))
tbl_benchmark_bold %>%
  kable(escape = FALSE, align = "c") %>%
  kable_styling(full_width = TRUE) %>%
  save_kable("analysis/results/benchmark_summary_table.html")
# <U+2500><U+2500> Render table <U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500><U+2500>
saveRDS(tbl_benchmark, "analysis/results/benchmark_summary_table.rds")

