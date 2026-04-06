library(svmodt)

australian_credit <- read.table("data/statlog-australian-credit_R.dat") |>
  mutate(clase = as.factor(clase)) |> standard_scaler()

set.seed(65)

split <- rsample::initial_split(data = australian_credit, prop = 0.7, strata = clase)

train <- training(split)
test <- testing(split)

svm_aus <- svm_split(data = train, 
                     response = "clase", 
                     min_samples = 10,
                     feature_method = "mutual", 
                     max_features = 3,
                     max_depth = 5,
                     class_weights = "balanced",
#                     max_features_strategy = "random",
                     #class_weights = "custom", 
                     #custom_class_weights = c("0" = 2, "1" = 1)
                     )

#Insample Acc
mean(svm_predict_tree(tree = svm_aus, newdata = train) == train$clase)

#Outsample Acc
mean(svm_predict_tree(tree = svm_aus, newdata = test) == test$clase)



australian_credit |>
  GGally::ggpairs()
