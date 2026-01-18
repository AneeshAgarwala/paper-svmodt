# Libraries
library(rsample)
library(dplyr)
library(purrr)
library(reticulate)
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
    cost = 1,
    verbose = FALSE,
    max_features = NULL,
    max_depth = 10
  )
}

## Default SVMODT
train_svmodt_default <- function(data, response){
  svmodt::svm_split(
    data = data,
    response = response,
    max_depth = 10,
    feature_method = "mutual")
}

# Default Python STree
train_python_stree <- function(data, response) {
  
  y <- data[[response]]
  X <- data |> dplyr::select(-all_of(response))
  
  py_model <- do.call(stree$Stree, stree_args)
  py_model$fit(X,y)
  
  return(py_model)
}

# Optimized STree for each dataset
train_stree_wdbc <- function(data, response) {
  stree_split(
    data = data,
    response = response,
    kernel = "linear",
    impurity_measure = "entropy",
    max_depth = 10,
    cost = 1,
    verbose = FALSE,
    max_features = NULL
  )
}

train_stree_iris <- function(data, response) {
  stree_split(
    data = data,
    response = response,
    kernel = "linear",
    impurity_measure = "entropy",
    max_depth = 10,
    cost = 1,
    verbose = FALSE,
    max_features = NULL
  )
}

train_stree_echocardiogram <- function(data, response) {
  stree_split(
    data = data,
    response = response,
    kernel = "polynomial",
    gamma = 0.1,
    impurity_measure = "entropy",
    max_depth = 10,
    cost = 7,
    max_features = "auto",
    verbose = FALSE
  )
}

train_stree_fertility <- function(data, response) {
  stree_split(
    data = data,
    response = response,
    kernel = "linear",
    impurity_measure = "entropy",
    max_depth = 10,
    cost = 0.05,
    max_features = "auto",
    verbose = FALSE
  )
}

train_stree_wine <- function(data, response) {
  stree_split(
    data = data,
    response = response,
    impurity_measure = "entropy",
    max_depth = 10,
    cost = 0.55,
    verbose = FALSE,
    max_features = NULL
  )
}

train_stree_ctg3 <- function(data, response) {
  stree_split(
    data = data,
    response = response,
    kernel = "linear",
    impurity_measure = "entropy",
    max_depth = 10,
    cost = 1,
    verbose = FALSE,
    max_features = NULL
  )
}

train_stree_ctg10 <- function(data, response) {
  stree_split(
    data = data,
    response = response,
    kernel = "linear",
    impurity_measure = "entropy",
    max_depth = 10,
    cost = 1,
    verbose = FALSE,
    max_features = NULL
  )
}

train_stree_ionosphere <- function(data, response) {
  stree_split(
    data = data,
    response = response,
    kernel = "polynomial",
    gamma = 0.1,
    impurity_measure = "entropy",
    max_depth = 10,
    cost = 7,
    max_features = "auto",
    verbose = FALSE
  )
}

train_stree_dermatology <- function(data, response) {
  stree_split(
    data = data,
    response = response,
    kernel = "linear",
    impurity_measure = "entropy",
    max_depth = 10,
    cost = 55, ## upper limit on cost in e1071::svm()
    verbose = FALSE,
    max_features = NULL
  )
}

train_stree_aus_credit <- function(data, response) {
  stree_split(
    data = data,
    response = response,
    kernel = "linear",
    impurity_measure = "entropy",
    max_depth = 10,
    cost = 0.05,
    max_features = "auto",
    verbose = FALSE
  )
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

# RUN 1: DEFAULT ARGUMENTS
stat_wdbc <- rep(NA, 10)
stat_iris <- rep(NA, 10)
stat_echocardiogram <- rep(NA, 10)
stat_fertility <- rep(NA, 10)
stat_wine <- rep(NA, 10)
stat_ctg3 <- rep(NA, 10)
stat_ctg10 <- rep(NA, 10)
stat_ionosphere <- rep(NA, 10)
stat_dermatology <- rep(NA, 10)
stat_aus_credit <- rep(NA, 10)
seed_list <- c(57, 31, 1714, 17, 23, 79, 83, 97, 7, 1)


for(i in 1:10){
  cat("Iteration", i, "- Default\n")
  
  set.seed(seed_list[i])
  ## WDBC
  cv_wdbc <- run_kfold_cv(
    data = wdbc,
    response = "clase",
    k = 5,
    train_fun = train_stree_default,
    predict_fun = predict_stree
  )
  stat_wdbc[i] <- mean(cv_wdbc$accuracy)
  
  ## IRIS
  cv_iris <- run_kfold_cv(
    data = iris,
    response = "clase",
    k = 5,
    train_fun = train_stree_default,
    predict_fun = predict_stree
  )
  stat_iris[i] <- mean(cv_iris$accuracy)
  
  ## ECHOCARDIOGRAM
  cv_echo <- run_kfold_cv(
    data = echocardiogram,
    response = "clase",
    k = 5,
    train_fun = train_stree_default,
    predict_fun = predict_stree
  )
  stat_echocardiogram[i] <- mean(cv_echo$accuracy)
  
  ## FERTILITY
  cv_fert <- run_kfold_cv(
    data = fertility,
    response = "clase",
    k = 5,
    train_fun = train_stree_default,
    predict_fun = predict_stree
  )
  stat_fertility[i] <- mean(cv_fert$accuracy)
  
  ## WINE
  cv_wine <- run_kfold_cv(
    data = wine,
    response = "clase",
    k = 5,
    train_fun = train_stree_default,
    predict_fun = predict_stree
  )
  stat_wine[i] <- mean(cv_wine$accuracy)
  
  ## CTG3
  cv_ctg3 <- run_kfold_cv(
    data = ctg3,
    response = "clase",
    k = 5,
    train_fun = train_stree_default,
    predict_fun = predict_stree
  )
  stat_ctg3[i] <- mean(cv_ctg3$accuracy)
  
  ## CTG10
  cv_ctg10 <- run_kfold_cv(
    data = ctg10,
    response = "clase",
    k = 5,
    train_fun = train_stree_default,
    predict_fun = predict_stree
  )
  stat_ctg10[i] <- mean(cv_ctg10$accuracy)
  
  ## IONOSPHERE
  cv_ionosphere <- run_kfold_cv(
    data = ionosphere,
    response = "clase",
    k = 5,
    train_fun = train_stree_default,
    predict_fun = predict_stree
  )
  stat_ionosphere[i] <- mean(cv_ionosphere$accuracy)
  
  ## DERMATOLOGY
  cv_dermatology <- run_kfold_cv(
    data = dermatology,
    response = "clase",
    k = 5,
    train_fun = train_stree_default,
    predict_fun = predict_stree
  )
  stat_dermatology[i] <- mean(cv_dermatology$accuracy)
  
  ## AUSTRALIAN CREDIT
  cv_aus_credit <- run_kfold_cv(
    data = australian_credit,
    response = "clase",
    k = 5,
    train_fun = train_stree_default,
    predict_fun = predict_stree
  )
  stat_aus_credit[i] <- mean(cv_aus_credit$accuracy)
}

# Results with default arguments
cat("\n========== DEFAULT ARGUMENTS RESULTS ==========\n")
cat("WDBC:           ", round(mean(stat_wdbc), 4), "\n")
cat("Iris:           ", round(mean(stat_iris), 4), "\n")
cat("Echocardiogram: ", round(mean(stat_echocardiogram), 4), "\n")
cat("Fertility:      ", round(mean(stat_fertility), 4), "\n")
cat("Wine:           ", round(mean(stat_wine), 4), "\n")
cat("CTG3:           ", round(mean(stat_ctg3), 4), "\n")
cat("CTG10:          ", round(mean(stat_ctg10), 4), "\n")
cat("Ionosphere:     ", round(mean(stat_ionosphere), 4), "\n")
cat("Dermatology:    ", round(mean(stat_dermatology), 4), "\n")
cat("Aus Credit:     ", round(mean(stat_aus_credit), 4), "\n")


# RUN2: SVMODT ALGORITHM
stat_svmodt_wdbc <- rep(NA, 10)
stat_svmodt_iris <- rep(NA, 10)
stat_svmodt_echocardiogram <- rep(NA, 10)
stat_svmodt_fertility <- rep(NA, 10)
stat_svmodt_wine <- rep(NA, 10)
stat_svmodt_ctg3 <- rep(NA, 10)
stat_svmodt_ctg10 <- rep(NA, 10)
stat_svmodt_ionosphere <- rep(NA, 10)
stat_svmodt_dermatology <- rep(NA, 10)
stat_svmodt_aus_credit <- rep(NA, 10)

for(i in 1:10){
  cat("Iteration", i, "- Default\n")
  
  set.seed(seed_list[i])
  ## WDBC
  cv_wdbc <- run_kfold_cv(
    data = wdbc,
    response = "clase",
    k = 5,
    train_fun = train_svmodt_default,
    predict_fun = predict_svmodt
  )
  stat_svmodt_wdbc[i] <- mean(cv_wdbc$accuracy)
  
  ## IRIS
  cv_iris <- run_kfold_cv(
    data = iris,
    response = "clase",
    k = 5,
    train_fun = train_svmodt_default,
    predict_fun = predict_svmodt
  )
  stat_svmodt_iris[i] <- mean(cv_iris$accuracy)
  
  ## ECHOCARDIOGRAM
  cv_echo <- run_kfold_cv(
    data = echocardiogram,
    response = "clase",
    k = 5,
    train_fun = train_svmodt_default,
    predict_fun = predict_svmodt
  )
  stat_svmodt_echocardiogram[i] <- mean(cv_echo$accuracy)
  
  ## FERTILITY
  cv_fert <- run_kfold_cv(
    data = fertility,
    response = "clase",
    k = 5,
    train_fun = train_svmodt_default,
    predict_fun = predict_svmodt
  )
  stat_svmodt_fertility[i] <- mean(cv_fert$accuracy)
  
  ## WINE
  cv_wine <- run_kfold_cv(
    data = wine,
    response = "clase",
    k = 5,
    train_fun = train_svmodt_default,
    predict_fun = predict_svmodt
  )
  stat_svmodt_wine[i] <- mean(cv_wine$accuracy)
  
  ## CTG3
  cv_ctg3 <- run_kfold_cv(
    data = ctg3,
    response = "clase",
    k = 5,
    train_fun = train_svmodt_default,
    predict_fun = predict_svmodt
  )
  stat_svmodt_ctg3[i] <- mean(cv_ctg3$accuracy)
  
  ## CTG10
  cv_ctg10 <- run_kfold_cv(
    data = ctg10,
    response = "clase",
    k = 5,
    train_fun = train_svmodt_default,
    predict_fun = predict_svmodt
  )
  stat_svmodt_ctg10[i] <- mean(cv_ctg10$accuracy)
  
  ## IONOSPHERE
  cv_ionosphere <- run_kfold_cv(
    data = ionosphere,
    response = "clase",
    k = 5,
    train_fun = train_svmodt_default,
    predict_fun = predict_svmodt
  )
  stat_svmodt_ionosphere[i] <- mean(cv_ionosphere$accuracy)
  
  ## DERMATOLOGY
  cv_dermatology <- run_kfold_cv(
    data = dermatology,
    response = "clase",
    k = 5,
    train_fun = train_svmodt_default,
    predict_fun = predict_svmodt
  )
  stat_svmodt_dermatology[i] <- mean(cv_dermatology$accuracy)
  
  ## AUSTRALIAN CREDIT
  cv_aus_credit <- run_kfold_cv(
    data = australian_credit,
    response = "clase",
    k = 5,
    train_fun = train_svmodt_default,
    predict_fun = predict_svmodt
  )
  stat_svmodt_aus_credit[i] <- mean(cv_aus_credit$accuracy)
}

# Results with default arguments
cat("\n========== DEFAULT ARGUMENTS RESULTS ==========\n")
cat("WDBC:           ", round(mean(stat_svmodt_wdbc), 4), "\n")
cat("Iris:           ", round(mean(stat_svmodt_iris), 4), "\n")
cat("Echocardiogram: ", round(mean(stat_svmodt_echocardiogram), 4), "\n")
cat("Fertility:      ", round(mean(stat_svmodt_fertility), 4), "\n")
cat("Wine:           ", round(mean(stat_svmodt_wine), 4), "\n")
cat("CTG3:           ", round(mean(stat_svmodt_ctg3), 4), "\n")
cat("CTG10:          ", round(mean(stat_svmodt_ctg10), 4), "\n")
cat("Ionosphere:     ", round(mean(stat_svmodt_ionosphere), 4), "\n")
cat("Dermatology:    ", round(mean(stat_svmodt_dermatology), 4), "\n")
cat("Aus Credit:     ", round(mean(stat_svmodt_aus_credit), 4), "\n")


# RUN 3: OPTIMIZED ARGUMENTS
cat("\n========== RUNNING WITH OPTIMIZED ARGUMENTS ==========\n\n")

opt_stat_wdbc <- rep(NA, 10)
opt_stat_iris <- rep(NA, 10)
opt_stat_echocardiogram <- rep(NA, 10)
opt_stat_fertility <- rep(NA, 10)
opt_stat_wine <- rep(NA, 10)
opt_stat_ctg3 <- rep(NA, 10)
opt_stat_ctg10 <- rep(NA, 10)
opt_stat_ionosphere <- rep(NA, 10)
opt_stat_dermatology <- rep(NA, 10)
opt_stat_aus_credit <- rep(NA, 10)

for(i in 1:10){
  cat("Iteration", i, "- Optimized\n")
  
  set.seed(seed_list[i])
  ## WDBC
  cv_wdbc <- run_kfold_cv(
    data = wdbc,
    response = "clase",
    k = 5,
    train_fun = train_stree_wdbc,
    predict_fun = predict_stree
  )
  opt_stat_wdbc[i] <- mean(cv_wdbc$accuracy)
  
  ## IRIS
  cv_iris <- run_kfold_cv(
    data = iris,
    response = "clase",
    k = 5,
    train_fun = train_stree_iris,
    predict_fun = predict_stree
  )
  opt_stat_iris[i] <- mean(cv_iris$accuracy)
  
  ## ECHOCARDIOGRAM
  cv_echo <- run_kfold_cv(
    data = echocardiogram,
    response = "clase",
    k = 5,
    train_fun = train_stree_echocardiogram,
    predict_fun = predict_stree
  )
  opt_stat_echocardiogram[i] <- mean(cv_echo$accuracy)
  
  ## FERTILITY
  cv_fert <- run_kfold_cv(
    data = fertility,
    response = "clase",
    k = 5,
    train_fun = train_stree_fertility,
    predict_fun = predict_stree
  )
  opt_stat_fertility[i] <- mean(cv_fert$accuracy)
  
  ## WINE
  cv_wine <- run_kfold_cv(
    data = wine,
    response = "clase",
    k = 5,
    train_fun = train_stree_wine,
    predict_fun = predict_stree
  )
  opt_stat_wine[i] <- mean(cv_wine$accuracy)
  
  ## CTG3
  cv_ctg3 <- run_kfold_cv(
    data = ctg3,
    response = "clase",
    k = 5,
    train_fun = train_stree_ctg3,
    predict_fun = predict_stree
  )
  opt_stat_ctg3[i] <- mean(cv_ctg3$accuracy)
  
  ## CTG10
  cv_ctg10 <- run_kfold_cv(
    data = ctg10,
    response = "clase",
    k = 5,
    train_fun = train_stree_ctg10,
    predict_fun = predict_stree
  )
  opt_stat_ctg10[i] <- mean(cv_ctg10$accuracy)
  
  ## IONOSPHERE
  cv_ionosphere <- run_kfold_cv(
    data = ionosphere,
    response = "clase",
    k = 5,
    train_fun = train_stree_ionosphere,
    predict_fun = predict_stree
  )
  opt_stat_ionosphere[i] <- mean(cv_ionosphere$accuracy)
  
  ## DERMATOLOGY
  cv_dermatology <- run_kfold_cv(
    data = dermatology,
    response = "clase",
    k = 5,
    train_fun = train_stree_dermatology,
    predict_fun = predict_stree
  )
  opt_stat_dermatology[i] <- mean(cv_dermatology$accuracy)
  
  ## AUSTRALIAN CREDIT
  cv_aus_credit <- run_kfold_cv(
    data = australian_credit,
    response = "clase",
    k = 5,
    train_fun = train_stree_aus_credit,
    predict_fun = predict_stree
  )
  opt_stat_aus_credit[i] <- mean(cv_aus_credit$accuracy)
}

# Results with optimized arguments
cat("\n========== OPTIMIZED ARGUMENTS RESULTS ==========\n")
cat("WDBC:           ", round(mean(opt_stat_wdbc), 4), "\n")
cat("Iris:           ", round(mean(opt_stat_iris), 4), "\n")
cat("Echocardiogram: ", round(mean(opt_stat_echocardiogram), 4), "\n")
cat("Fertility:      ", round(mean(opt_stat_fertility), 4), "\n")
cat("Wine:           ", round(mean(opt_stat_wine), 4), "\n")
cat("CTG3:           ", round(mean(opt_stat_ctg3), 4), "\n")
cat("CTG10:          ", round(mean(opt_stat_ctg10), 4), "\n")
cat("Ionosphere:     ", round(mean(opt_stat_ionosphere), 4), "\n")
cat("Dermatology:    ", round(mean(opt_stat_dermatology), 4), "\n")
cat("Aus Credit:     ", round(mean(opt_stat_aus_credit), 4), "\n")


# RUN4: PYTHON STREE WITH RETICULATE
py_stree_wdbc <- rep(NA, 10)
py_stree_iris <- rep(NA, 10)
py_stree_echocardiogram <- rep(NA, 10)
py_stree_fertility <- rep(NA, 10)
py_stree_wine <- rep(NA, 10)
py_stree_ctg3 <- rep(NA, 10)
py_stree_ctg10 <- rep(NA, 10)
py_stree_ionosphere <- rep(NA, 10)
py_stree_dermatology <- rep(NA, 10)
py_stree_aus_credit <- rep(NA, 10)

for(i in 1:10){
  cat("Iteration", i, "- Default\n")
  
  set.seed(seed_list[i])
  ## WDBC
  cv_wdbc <- run_kfold_cv(
    data = wdbc,
    response = "clase",
    k = 5,
    train_fun = train_python_stree,
    predict_fun = predict_python_stree
  )
  py_stree_wdbc[i] <- mean(cv_wdbc$accuracy)
  
  ## IRIS
  cv_iris <- run_kfold_cv(
    data = iris,
    response = "clase",
    k = 5,
    train_fun = train_python_stree,
    predict_fun = predict_python_stree
  )
  py_stree_iris[i] <- mean(cv_iris$accuracy)
  
  ## ECHOCARDIOGRAM
  cv_echo <- run_kfold_cv(
    data = echocardiogram,
    response = "clase",
    k = 5,
    train_fun = train_python_stree,
    predict_fun = predict_python_stree
  )
  py_stree_echocardiogram[i] <- mean(cv_echo$accuracy)
  
  ## FERTILITY
  cv_fert <- run_kfold_cv(
    data = fertility,
    response = "clase",
    k = 5,
    train_fun = train_python_stree,
    predict_fun = predict_python_stree
  )
  py_stree_fertility[i] <- mean(cv_fert$accuracy)
  
  ## WINE
  cv_wine <- run_kfold_cv(
    data = wine,
    response = "clase",
    k = 5,
    train_fun = train_python_stree,
    predict_fun = predict_python_stree
  )
  py_stree_wine[i] <- mean(cv_wine$accuracy)
  
  ## CTG3
  cv_ctg3 <- run_kfold_cv(
    data = ctg3,
    response = "clase",
    k = 5,
    train_fun = train_python_stree,
    predict_fun = predict_python_stree
  )
  py_stree_ctg3[i] <- mean(cv_ctg3$accuracy)
  
  ## CTG10
  cv_ctg10 <- run_kfold_cv(
    data = ctg10,
    response = "clase",
    k = 5,
    train_fun = train_python_stree,
    predict_fun = predict_python_stree
  )
  py_stree_ctg10[i] <- mean(cv_ctg10$accuracy)
  
  ## IONOSPHERE
  cv_ionosphere <- run_kfold_cv(
    data = ionosphere,
    response = "clase",
    k = 5,
    train_fun = train_python_stree,
    predict_fun = predict_python_stree
  )
  py_stree_ionosphere[i] <- mean(cv_ionosphere$accuracy)
  
  ## DERMATOLOGY
  cv_dermatology <- run_kfold_cv(
    data = dermatology,
    response = "clase",
    k = 5,
    train_fun = train_python_stree,
    predict_fun = predict_python_stree
  )
  py_stree_dermatology[i] <- mean(cv_dermatology$accuracy)
  
  ## AUSTRALIAN CREDIT
  cv_aus_credit <- run_kfold_cv(
    data = australian_credit,
    response = "clase",
    k = 5,
    train_fun = train_python_stree,
    predict_fun = predict_python_stree
  )
  py_stree_aus_credit[i] <- mean(cv_aus_credit$accuracy)
}


# Results with Reticulate Python Stree
cat("\n========== OPTIMIZED ARGUMENTS RESULTS ==========\n")
cat("WDBC:           ", round(mean(py_stree_wdbc), 4), "\n")
cat("Iris:           ", round(mean(py_stree_iris), 4), "\n")
cat("Echocardiogram: ", round(mean(py_stree_echocardiogram), 4), "\n")
cat("Fertility:      ", round(mean(py_stree_fertility), 4), "\n")
cat("Wine:           ", round(mean(py_stree_wine), 4), "\n")
cat("CTG3:           ", round(mean(py_stree_ctg3), 4), "\n")
cat("CTG10:          ", round(mean(py_stree_ctg10), 4), "\n")
cat("Ionosphere:     ", round(mean(py_stree_ionosphere), 4), "\n")
cat("Dermatology:    ", round(mean(py_stree_dermatology), 4), "\n")
cat("Aus Credit:     ", round(mean(py_stree_aus_credit), 4), "\n")

