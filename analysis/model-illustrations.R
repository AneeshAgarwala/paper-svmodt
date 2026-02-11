library(rsample)
library(dplyr)
library(purrr)
library(reticulate)
#install.packages("D:/SVMODT/project-svodt/", repos = NULL, type = "source")
library(svmodt)
# Libraries - Python
stree <- import("stree")
sklearn_svm <- import("sklearn.svm")


penguins_data <- penguins |>
  mutate(island = as.numeric(island),
         sex = as.numeric(sex),
         year = as.numeric(year)) |>
  na.omit()



svmodt_tree <- svmodt::svm_split(data = penguins_data, 
                                 response = "species", 
                                 verbose = FALSE, 
                                 max_features = 3, 
                                 feature_method = "random", n_subsets = 30, max_depth = 20)

svmodt_preds <- svmodt::svm_predict_tree(tree = svmodt_tree, newdata = penguins_data)

svmodt::print_svm_tree(svmodt_tree, show_probabilities = TRUE, show_penalties = FALSE)

table(penguins_data$species == svmodt_preds)





stree_tree <- svmodt::stree_split(data = penguins_data, response = "species", max_depth = 10, kernel = "linear", verbose = TRUE, cost = 1)
stree_preds <- svmodt::stree_predict(tree = stree_tree, newdata = penguins_data)

table(penguins_data$species == stree_preds)






