# Libraries
library(dplyr)
library(purrr)
library(svmodt)


# ── Datasets ──────────────────────────────────────────────────────────────────
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



class_imbalance_ratio <- numeric(length(datasets))
names(class_imbalance_ratio) <- names(datasets)

for (name in names(datasets)) {
  class_freq <- table(datasets[[name]]$clase)
  class_imbalance_ratio[name] <- min(class_freq) / max(class_freq)
}

class_imbalance_ratio |> saveRDS("analysis/results/class-imbalance-ratio.RDS")



