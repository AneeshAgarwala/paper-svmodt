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

train_python_stree <- function(data, response) {
  
  y <- data[[response]]
  X <- data |> dplyr::select(-all_of(response))
  
  py_model <- do.call(stree$Stree, stree_args)
  py_model$fit(X,y)
  
  return(py_model)
}



bench::mark(
    train_stree_default(wdbc, "clase"),
    train_python_stree(wdbc, "clase"), check = FALSE
  )$median -> wdbc_run

bench::mark(
  train_stree_default(ctg3, "clase"),
  train_python_stree(ctg3, "clase"), check = FALSE
)$median -> ctg3_run

bench::mark(
  train_stree_default(ctg10, "clase"),
  train_python_stree(ctg10, "clase"), check = FALSE
)$median -> ctg10_run

bench::mark(
  train_stree_default(iris, "clase"),
  train_python_stree(iris, "clase"), check = FALSE
)$median -> iris_run

bench::mark(
  train_stree_default(dermatology, "clase"),
  train_python_stree(dermatology, "clase"), check = FALSE
)$median -> dermatology_run

bench::mark(
  train_stree_default(echocardiogram, "clase"),
  train_python_stree(echocardiogram, "clase"), check = FALSE
)$median -> echocardiogram_run

bench::mark(
  train_stree_default(fertility, "clase"),
  train_python_stree(fertility, "clase"), check = FALSE
)$median -> fertility_run

bench::mark(
  train_stree_default(ionosphere, "clase"),
  train_python_stree(ionosphere, "clase"), check = FALSE
)$median -> ionosphere_run

bench::mark(
  train_stree_default(australian_credit, "clase"),
  train_python_stree(australian_credit, "clase"), check = FALSE
)$median -> australian_credit_run

bench::mark(
  train_stree_default(wine, "clase"),
  train_python_stree(wine, "clase"), check = FALSE
)$median -> wine_run

time_bench <- rbind(wdbc_run, iris_run, echocardiogram_run, fertility_run, wine_run, ctg3_run, ctg10_run, ionosphere_run, dermatology_run, australian_credit_run)
colnames(time_bench) <- c("StreeR", "Stree")
time_bench |> saveRDS(file = "analysis/results/time-benchmark.rds")
