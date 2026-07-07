library(rsample)
library(dplyr)
library(tidyr)
library(purrr)
library(svmodt)
library(tourr)

### Data
set.seed(235)
penguins_orsf <- penguins |>
  select(-island, -sex, -year) |>
  drop_na()

penguins_split <- rsample::initial_split(data = penguins_orsf, strata = species, prop = 0.6)
penguins_train <- training(penguins_split)
penguins_test <- testing(penguins_split)



# Generate grid spanning the scaled feature space
feature_cols <- colnames(penguins_train[, 2:5])

# Get min/max of each feature from training data to stay within bounds
feature_ranges <- penguins_train |>
  select(all_of(feature_cols)) |>
  summarise(across(everything(), list(min = min, max = max)))

n_points <- 5000 # more points = smoother boundaries

set.seed(235)

mass_points <- data.frame(
  species = NA,
  bill_length_mm = runif(n_points, feature_ranges$bill_length_mm_min, feature_ranges$bill_length_mm_max),
  bill_depth_mm = runif(n_points, feature_ranges$bill_depth_mm_min, feature_ranges$bill_depth_mm_max),
  flipper_length_mm = runif(n_points, feature_ranges$flipper_length_mm_min, feature_ranges$flipper_length_mm_max),
  body_mass_g = runif(n_points, feature_ranges$body_mass_g_min, feature_ranges$body_mass_g_max)
)

colnames(mass_points) <- colnames(penguins_test)

####### FEATURE SELECTION #########


tour_path <- save_history(mass_points[, 2:5], little_tour())


## Random Feature
fit_random <- svm_split(
  data = penguins_train,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  feature_method = "random", n_subsets = 10,
  max_features = 3
)

preds_random <- predict(fit_random, mass_points)
preds_random <- as.factor(preds_random)


render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_random,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-random.gif"
)

## Mutual Features
fit_mutual <- svm_split(
  data = penguins_train,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  max_features = 3,
  feature_method = "mutual"
)

preds_mutual <- predict(fit_mutual, mass_points)
preds_mutual <- as.factor(preds_mutual)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_mutual,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-mutual.gif"
)


## Correlated Features
fit_cor <- svm_split(
  data = penguins_train,
  response = "species",
  max_depth = 3,
  min_samples = 5,
  max_features = 3,
  feature_method = "cor"
)

preds_corr <- predict(fit_cor, mass_points)
preds_corr <- as.factor(preds_corr)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_corr,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-corr.gif"
)

####### CLASS WEIGHTS #########

## NONE ##
fit_none <- svm_split(
  data = penguins_train,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  feature_method = "mutual",
  max_features = 3, class_weights = "none"
)

preds_none <- predict(fit_none, mass_points)
preds_none <- as.factor(preds_none)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_none,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-none.gif"
)

## BALANCED ##
fit_balanced <- svm_split(
  data = penguins_train,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  feature_method = "mutual",
  max_features = 3, class_weights = "balanced"
)

preds_balanced <- predict(fit_balanced, mass_points)
preds_balanced <- as.factor(preds_balanced)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_balanced,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-balanced.gif"
)


## CUSTOM ##
custom_weights <- c("Adelie" = 2, "Gentoo" = 1, "Chinstrap" = 5)

fit_custom <- svm_split(
  data = penguins_train,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  feature_method = "mutual",
  max_features = 3, class_weights = "custom", custom_class_weights = custom_weights
)


preds_custom <- predict(fit_custom, mass_points)
preds_custom <- as.factor(preds_custom)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_custom,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-custom.gif"
)

####### MAX FEATURE SELECTION #########

## Constant Max Feature
fit_constant <- svm_split(
  data = penguins_train,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  feature_method = "mutual",
  max_features_strategy = "constant"
)

preds_constant <- predict(fit_constant, mass_points)
preds_constant <- as.factor(preds_constant)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_constant,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-constant.gif"
)


## Decreasing Max Features
fit_decrease <- svm_split(
  data = penguins_train,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  feature_method = "mutual",
  max_features_strategy = "decrease",
  max_features_decrease_rate = 0.5
)

preds_decrease <- predict(fit_decrease, mass_points)
preds_decrease <- as.factor(preds_decrease)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_decrease,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-decrease.gif"
)


## Random Max Features
fit_max_random <- svm_split(
  data = penguins_train,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  feature_method = "mutual",
  max_features_strategy = "random",
  max_features_random_range = c(0.5, 0.8)
)

preds_max_random <- predict(fit_random, mass_points)
preds_max_random <- as.factor(preds_max_random)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_max_random,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-random-maxfeat.gif"
)



####### MAX FEATURE SELECTION #########

## No Penalty
fit_no_penalty <- svm_split(
  data = penguins_orsf,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  max_features = 2,
  feature_method = "mutual",
  penalize_used_features = FALSE
)

preds_no_penalty <- predict(fit_no_penalty, mass_points)
preds_no_penalty <- as.factor(preds_no_penalty)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_no_penalty,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-no-penalty.gif"
)

## Low Penalty
fit_low_penalty <- svm_split(
  data = penguins_orsf,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  max_features = 2,
  feature_method = "mutual",
  penalize_used_features = TRUE,
  feature_penalty_weight = 0.2
)

preds_low_penalty <- predict(fit_low_penalty, mass_points)
preds_low_penalty <- as.factor(preds_low_penalty)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_low_penalty,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-low-penalty.gif"
)

## Medium Penalty
fit_medium_penalty <- svm_split(
  data = penguins_orsf,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  max_features = 2,
  feature_method = "mutual",
  penalize_used_features = TRUE,
  feature_penalty_weight = 0.5
)

preds_medium_penalty <- predict(fit_medium_penalty, mass_points)
preds_medium_penalty <- as.factor(preds_medium_penalty)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_medium_penalty,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-medium-penalty.gif"
)


## High Penalty
fit_high_penalty <- svm_split(
  data = penguins_orsf,
  response = "species",
  max_depth = 4,
  min_samples = 5,
  max_features = 2,
  feature_method = "mutual",
  penalize_used_features = TRUE,
  feature_penalty_weight = 0.8
)

preds_high_penalty <- predict(fit_high_penalty, mass_points)
preds_high_penalty <- as.factor(preds_high_penalty)

render_gif(
  data = mass_points[, 2:5],
  tour_path = planned_tour(tour_path),
  rescale = TRUE, frames = 200,
  display = display_xy(
    col = preds_high_penalty,
    center = FALSE,
    half_range = 2.5
  ),
  gif_file = "figures/tour-high-penalty.gif"
)
