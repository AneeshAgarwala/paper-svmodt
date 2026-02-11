# Libraries
library(rsample)
library(dplyr)
library(purrr)
library(reticulate)
#devtools::load_all()
#install.packages("D:/SVMODT/project-svodt/", repos = NULL, type = "source")
library(svmodt)
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


## Benchmarked Datasets
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


# Generic 5-fold CV function
run_kfold_cv <- function(data, response, k = 5, train_fun, predict_fun) {
  
  # Create folds
  folds <- vfold_cv(data, v = k, strata = !!sym(response))
  
  # Evaluate each fold
  results <- map_df(folds$splits, function(split) {
    
    train_data <- analysis(split)
    test_data  <- assessment(split)
    
    # Train model (user-defined)
    model <- train_fun(train_data, response)
    
    # Predict (user-defined)
    preds <- predict_fun(model, test_data, response)
    
    # Accuracy
    tibble(
      accuracy = mean(preds == test_data[[response]])
    )
  })
  
  return(results)
}

# Training functions

## Default STree
train_stree_default <- function(data, response) {
  stree_split(
    data = data,
    response = response,
    kernel = "linear",
    impurity_measure = "entropy",
    verbose = FALSE,
    max_depth = 15
  )
}

## Default SVMODT
train_svmodt_default <- function(data, response){
  svmodt::svm_split(
    data = data,
    response = response,
    max_depth = 15, 
    verbose = FALSE)
}

# Default Python STree
train_python_stree <- function(data, response) {
  
  y <- data[[response]]
  X <- data |> dplyr::select(-all_of(response))
  
  py_model <- do.call(stree$Stree, stree_args)
  py_model$fit(X,y)
  
  return(py_model)
}

# Prediction function
predict_stree <- function(model, newdata,...) {
  stree_predict(model, newdata)
}

predict_svmodt <- function(model, newdata, ...){
  svm_predict_tree(model, newdata)
}


predict_python_stree <- function(model, test_data, response) {
  
  X_test <- test_data |> dplyr::select(-all_of(response))
  
  preds <- model$predict(X_test)
  
  return(py_to_r(preds))
}

#### COMBINED RUN: ALL THREE ALGORITHMS WITH IDENTICAL SPLITS ####

# Initialize result storage
results <- list(
  r_stree = list(),
  r_svmodt = list(),
  py_stree = list()
)

datasets <- list(
  wdbc = wdbc,
  iris = iris,
  echocardiogram = echocardiogram,
  fertility = fertility,
  wine = wine,
  ctg3 = ctg3,
  ctg10 = ctg10,
  ionosphere = ionosphere,
  dermatology = dermatology,
  aus_credit = australian_credit
)

seed_list <- c(57, 31, 1714, 17, 23, 79, 83, 97, 7, 1)

for(i in 1:10){
  cat("Iteration", i, "\n")
  
  # Set seed once per iteration
  set.seed(seed_list[i])
  
  # Create folds once for all datasets
  folds_list <- map(datasets, ~vfold_cv(.x, v = 5, strata = clase))
  
  # Iterate through each dataset
  for(dataset_name in names(datasets)){
    cat("  Processing", dataset_name, "\n")
    
    # Use pre-created folds for this dataset
    current_folds <- folds_list[[dataset_name]]
    
    # Evaluate each fold for all three algorithms
    results_fold <- map_df(current_folds$splits, function(split) {
      
      train_data <- analysis(split)
      test_data  <- assessment(split)
      
      # R STree
      model_r_stree <- train_stree_default(train_data, "clase")
      preds_r_stree <- predict_stree(model_r_stree, test_data)
      acc_r_stree <- mean(preds_r_stree == test_data$clase)
      
      # R SVMODT
      model_r_svmodt <- train_svmodt_default(train_data, "clase")
      preds_r_svmodt <- predict_svmodt(model_r_svmodt, test_data)
      acc_r_svmodt <- mean(preds_r_svmodt == test_data$clase)
      
      # Python STree
      model_py_stree <- train_python_stree(train_data, "clase")
      preds_py_stree <- predict_python_stree(model_py_stree, test_data, "clase")
      acc_py_stree <- mean(preds_py_stree == test_data$clase)
      
      tibble(
        r_stree = acc_r_stree,
        r_svmodt = acc_r_svmodt,
        py_stree = acc_py_stree
      )
    })
    
    # Store mean accuracies
    results$r_stree[[dataset_name]][i] <- mean(results_fold$r_stree)
    results$r_svmodt[[dataset_name]][i] <- mean(results_fold$r_svmodt)
    results$py_stree[[dataset_name]][i] <- mean(results_fold$py_stree)
  }
}

# Convert to matrices (same format as your original code)
r_stree <- do.call(cbind, results$r_stree)
colnames(r_stree) <- c("stat_wdbc", "stat_iris", "stat_echocardiogram", 
                       "stat_fertility", "stat_wine", "stat_ctg3", 
                       "stat_ctg10", "stat_ionosphere", "stat_dermatology", 
                       "stat_aus_credit")

r_svmodt <- do.call(cbind, results$r_svmodt)
colnames(r_svmodt) <- c("stat_svmodt_wdbc", "stat_svmodt_iris", "stat_svmodt_echocardiogram",
                        "stat_svmodt_fertility", "stat_svmodt_wine", "stat_svmodt_ctg3",
                        "stat_svmodt_ctg10", "stat_svmodt_ionosphere", "stat_svmodt_dermatology",
                        "stat_svmodt_aus_credit")

py_stree <- do.call(cbind, results$py_stree)
colnames(py_stree) <- c("py_stree_wdbc", "py_stree_iris", "py_stree_echocardiogram",
                        "py_stree_fertility", "py_stree_wine", "py_stree_ctg3",
                        "py_stree_ctg10", "py_stree_ionosphere", "py_stree_dermatology",
                        "py_stree_aus_credit")

# Save results
r_stree |> saveRDS("analysis/results/r_stree.rds")
py_stree |> saveRDS("analysis/results/py_stree.rds")
r_svmodt |> saveRDS("analysis/results/r_svmodt.rds")


