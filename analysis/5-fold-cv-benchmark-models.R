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

# ── Python ─────────────────────────────────────────────────────────────────────
stree       <- import("stree")
sklearn_svm <- import("sklearn.svm")

# ── Datasets ───────────────────────────────────────────────────────────────────
ctg3 <- read.table("data/cardiotocography-3clases_R.dat") |>
  mutate(clase = as.factor(clase)) |> standard_scaler()

ctg10 <- read.table("data/cardiotocography-10clases_R.dat") |>
  mutate(clase = as.factor(clase)) |> standard_scaler()

australian_credit <- read.table("data/statlog-australian-credit_R.dat") |>
  mutate(clase = as.factor(clase)) |> standard_scaler()

wdbc <- read.table("data/breast-cancer-wisc-diag_R.dat") |>
  mutate(clase = as.factor(clase)) |> standard_scaler()

iris <- read.table("data/iris_R.dat") |>
  mutate(clase = as.factor(clase)) |> standard_scaler()

echocardiogram <- read.table("data/echocardiogram_R.dat") |>
  mutate(clase = as.factor(clase)) |> standard_scaler()

fertility <- read.table("data/fertility_R.dat") |>
  mutate(clase = as.factor(clase)) |> standard_scaler()

wine <- read.table("data/wine_R.dat") |>
  mutate(clase = as.factor(clase)) |> standard_scaler()

ionosphere <- read.table("data/ionosphere_R.dat") |>
  mutate(clase = as.factor(clase)) |> standard_scaler()

dermatology <- read.table("data/dermatology_R.dat") |>
  mutate(clase = as.factor(clase)) |> standard_scaler()

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

dataset_col_names  <- paste0("stat_", names(datasets))
seed_list          <- c(57, 31, 1714, 17, 23, 79, 83, 97, 7, 1)

data_names <- c(
  "WDBC Diagnosis", "Iris", "Echocardiogram", "Fertility", "Wine",
  "Cardiotography-3", "Cardiotography-10", "Ionosphere",
  "Dermatology")
data_num_observations <- c(569, 150, 131, 100, 178, 2126, 2126, 351, 366)
data_num_features     <- c(30, 4, 10, 9, 12, 21, 21, 33, 34)
data_num_classes      <- c(2, 3, 2, 2, 3, 3, 10, 2, 6)

# ── Python STree best configurations per dataset ───────────────────────────────
stree_best_args <- list(
  wdbc           = list(C = 0.2,    max_iter = 1e4L,       kernel = "linear"),
  iris           = list(            max_iter = 10000000L,  kernel = "linear"),
  echocardiogram = list(C = 7,      gamma = 0.1,           kernel = "poly",
                        max_features = "auto", max_iter = 1e4L),
  fertility      = list(C = 0.05,   max_features = "auto", max_iter = 1e4L,
                        kernel = "linear"),
  wine           = list(C = 0.55,   max_iter = 1e4L,       kernel = "linear"),
  ctg3           = list(            max_iter = 10000000L,  kernel = "linear"),
  ctg10          = list(            max_iter = 10000000L,  kernel = "linear"),
  ionosphere     = list(C = 7,      gamma = 0.1,           kernel = "rbf",
                        max_iter = 1e4L),
  dermatology    = list(C = 55,     max_iter = 1e4L,       kernel = "linear"),
  aus_credit     = list(C = 0.05,   max_features = "auto", max_iter = 1e4L,
                        kernel = "linear")
)

# ── Training Functions ─────────────────────────────────────────────────────────

train_rpart_shallow <- function(data, response) {
  f <- as.formula(paste(response, "~ ."))
  rpart::rpart(f, data = data,
               control = rpart::rpart.control(maxdepth = 3, cp = 0.01))
}

train_rpart_deep <- function(data, response) {
  f <- as.formula(paste(response, "~ ."))
  rpart::rpart(f, data = data,
               control = rpart::rpart.control(maxdepth = 15, cp = 0.001))
}

train_svm_linear <- function(data, response) {
  X <- data |> select(-all_of(response))
  y <- data[[response]]
  e1071::svm(x = X, y = y, kernel = "linear",
             scale = FALSE, probability = TRUE)
}

train_svm_rbf <- function(data, response) {
  X <- data |> select(-all_of(response))
  y <- data[[response]]
  e1071::svm(x = X, y = y, kernel = "radial",
             scale = FALSE, probability = TRUE)
}

train_aorsf <- function(data, response) {
  f <- as.formula(paste(response, "~ ."))
  aorsf::orsf(f, data = data, control = orsf_control_classification(), verbose_progress = FALSE)
}

train_logreg <- function(data, response) {
  lvls      <- levels(data[[response]])
  n_classes <- length(lvls)
  f         <- as.formula(paste(response, "~ ."))
  
  if (n_classes == 2) {
    data_glm             <- data
    data_glm[[response]] <- as.integer(data[[response]]) - 1L
    model                <- glm(f, data = data_glm, family = binomial())
    # Store training levels for safe prediction
    model$.response_levels <- lvls
    model
  } else {
    nnet::multinom(f, data = data, trace = FALSE, MaxNWts = 10000)
  }
}

# train_pptree <- function(data, response) {
#   PPtreeExt::PPtreeExtclass(
#     as.formula(paste(response, "~ .")),
#     data     = data,
#     PPmethod = "LDA", 
#     srule = FALSE, 
#     tol = 0.1
#   )
# }

train_python_stree_best <- function(data, response, dataset_name) {
  y        <- data[[response]]
  X        <- data |> dplyr::select(-all_of(response))
  args     <- c(list(tol = 0.0001), stree_best_args[[dataset_name]])
  py_model <- do.call(stree$Stree, args)
  py_model$fit(X, y)
  py_model
}

# ── Prediction Functions ───────────────────────────────────────────────────────

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
  
  message("    [WARN] predict_logreg: unknown model class: ",
          paste(class(model), collapse = ", "))
  rep(NA_character_, nrow(newdata))
}

# predict_pptree <- function(model, newdata, response) {
#   X <- newdata |> select(-all_of(response))
#     preds <- predict(object = model, X)
#     as.character(preds)
# }

predict_python_stree_best <- function(model, test_data, response) {
  X_test <- test_data |> dplyr::select(-all_of(response))
  preds  <- model$predict(X_test)
  as.character(py_to_r(preds))
}

# ── Benchmark Runner ───────────────────────────────────────────────────────────
run_benchmark <- function(datasets, seed_list, n_iter = 10) {
  
  model_names <- c(
    "rpart_shallow", "rpart_deep",
    "svm_linear",    "svm_rbf",
    "aorsf",         "logreg",
    "pptree",        "py_stree_best"
  )
  
  results <- map(model_names, ~list()) |> setNames(model_names)
  
  for (i in 1:n_iter) {
    cat("Iteration", i, "\n")
    
    set.seed(seed_list[i])
    folds_list <- map(datasets, ~vfold_cv(.x, v = 5, strata = clase))
    
    for (dataset_name in names(datasets)) {
      cat("  Processing", dataset_name, "\n")
      
      current_folds <- folds_list[[dataset_name]]
      
      results_fold <- map_df(current_folds$splits, function(split) {
        
        train_data <- analysis(split)
        test_data  <- assessment(split)
        truth      <- as.character(test_data$clase)
        
        # Generic safe accuracy wrapper with model name for diagnostics
        safe_acc <- function(train_fn, pred_fn, model_name, ...) {
          tryCatch({
            model <- train_fn(train_data, "clase", ...)
            preds <- pred_fn(model, test_data, "clase")
            mean(as.character(preds) == truth, na.rm = TRUE)
          }, error = function(e) {
            message("    [WARN] ", model_name, " | ", dataset_name,
                    ": ", conditionMessage(e))
            NA_real_
          })
        }
        
        tibble(
          rpart_shallow = safe_acc(train_rpart_shallow, predict_rpart,     "rpart_shallow"),
          rpart_deep    = safe_acc(train_rpart_deep,    predict_rpart,     "rpart_deep"),
          svm_linear    = safe_acc(train_svm_linear,    predict_svm_e1071, "svm_linear"),
          svm_rbf       = safe_acc(train_svm_rbf,       predict_svm_e1071, "svm_rbf"),
          aorsf         = safe_acc(train_aorsf,         predict_aorsf,     "aorsf"),
          logreg        = safe_acc(train_logreg,        predict_logreg,    "logreg"),
          #pptree        = safe_acc(train_pptree,        predict_pptree,    "pptree"),
          py_stree_best = tryCatch({
            model <- train_python_stree_best(train_data, "clase", dataset_name)
            preds <- predict_python_stree_best(model, test_data, "clase")
            mean(as.character(preds) == truth, na.rm = TRUE)
          }, error = function(e) {
            message("    [WARN] py_stree_best | ", dataset_name,
                    ": ", conditionMessage(e))
            NA_real_
          })
        )
      })
      
      for (model_name in model_names) {
        results[[model_name]][[dataset_name]][i] <- mean(
          results_fold[[model_name]], na.rm = TRUE
        )
      }
    }
  }
  
  return(results)
}

# ── Run ────────────────────────────────────────────────────────────────────────
cat("\n====== BENCHMARK: Multiple Classifiers ======\n")
benchmark_results <- run_benchmark(datasets, seed_list)

# ── Convert to Matrices ────────────────────────────────────────────────────────
benchmark_matrices <- map(benchmark_results, function(res) {
  mat <- do.call(cbind, res)
  colnames(mat) <- dataset_col_names
  mat
})

# ── Save ───────────────────────────────────────────────────────────────────────
iwalk(benchmark_matrices, function(mat, name) {
  saveRDS(mat, paste0("analysis/results/benchmark_", name, ".rds"))
})

cat("\nSaved", length(benchmark_matrices), "result matrices.\n")

# ── Summary Table ──────────────────────────────────────────────────────────────
# ── Load all results ───────────────────────────────────────────────────────────
r_svmodt_star <- readRDS("analysis/results/r_svmodt_star.rds")[,1:9]
r_svmodt <- readRDS("analysis/results/r_svmodt.rds")[,1:9]
logreg <- readRDS("analysis/results/benchmark_logreg.rds")[,1:9]
rpart_shallow <- readRDS("analysis/results/benchmark_rpart_shallow.rds")[,1:9]
rpart_deep <- readRDS("analysis/results/benchmark_rpart_deep.rds")[,1:9]
svm_linear <- readRDS("analysis/results/benchmark_svm_linear.rds")[,1:9]
py_stree_best <- readRDS("analysis/results/benchmark_py_stree_best.rds")[,1:9]

# ── Summary Table ──────────────────────────────────────────────────────────────
mean_sd <- function(x, digits = 3) {
  x <- as.numeric(x)
  sprintf(paste0("%.", digits, "f \u00B1 %.", digits, "f"),
          mean(x, na.rm = TRUE), sd(x, na.rm = TRUE))
}

tbl_benchmark <- data.frame(
  Dataset       = data_names,
  N             = data_num_observations,
  X             = data_num_features,
  L             = data_num_classes,
  rpart_shallow = apply(rpart_shallow, 2, mean_sd),
  rpart_deep    = apply(rpart_deep,    2, mean_sd),
  svm_linear    = apply(svm_linear,    2, mean_sd),
  #svm_rbf       = apply(benchmark_matrices$svm_rbf,       2, mean_sd),
  #aorsf         = apply(benchmark_matrices$aorsf,         2, mean_sd),
  logreg        = apply(logreg,        2, mean_sd),
  #pptree        = apply(benchmark_matrices$pptree,        2, mean_sd),
  py_stree_best = apply(py_stree_best, 2, mean_sd),
  svmodt_default = apply(as.matrix(r_svmodt), 2, mean_sd),
  svmodt_star   = apply(as.matrix(r_svmodt_star),         2, mean_sd),
  stringsAsFactors = FALSE
)

# ── Bold best per row ──────────────────────────────────────────────────────────
model_cols <- 5:ncol(tbl_benchmark)

tbl_benchmark_bold <- tbl_benchmark
tbl_benchmark_bold[model_cols] <- t(apply(tbl_benchmark[model_cols], 1, function(row) {
  means   <- as.numeric(sub(" \u00B1.*", "", row))
  max_idx <- which(means == max(means, na.rm = TRUE))
  out     <- row
  out[max_idx] <- cell_spec(out[max_idx], bold = TRUE)
  out
}))

# ── Render table ───────────────────────────────────────────────────────────────
saveRDS(tbl_benchmark, "analysis/results/benchmark_summary_table.rds")
