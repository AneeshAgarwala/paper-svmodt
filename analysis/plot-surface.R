plot_svmodt_surface <- function(tree, data, response, resolution = 200) {
  # Get the features used in the tree
  all_features <- get_tree_features(tree)

  if (length(all_features) < 2) {
    stop("Tree must use at least 2 features for surface plotting")
  }

  # Use first two features for the grid axes
  plot_features <- all_features[1:2]

  # Create grid for the two plotting dimensions
  grid <- expand.grid(
    x = seq(min(data[[plot_features[1]]], na.rm = TRUE),
      max(data[[plot_features[1]]], na.rm = TRUE),
      length.out = resolution
    ),
    y = seq(min(data[[plot_features[2]]], na.rm = TRUE),
      max(data[[plot_features[2]]], na.rm = TRUE),
      length.out = resolution
    )
  )

  names(grid) <- plot_features

  # CRITICAL FIX: Add ALL other features that might be used anywhere in the tree
  # Set them to their median values
  other_features <- setdiff(names(data), c(response, plot_features))
  for (feat in other_features) {
    if (is.numeric(data[[feat]])) {
      grid[[feat]] <- median(data[[feat]], na.rm = TRUE)
    } else if (is.factor(data[[feat]]) || is.character(data[[feat]])) {
      # For categorical features, use the mode (most common value)
      grid[[feat]] <- names(sort(table(data[[feat]]), decreasing = TRUE))[1]
    }
  }

  # IMPORTANT: Reorder columns to match original data structure
  # This ensures the scaler receives columns in the expected order
  grid <- grid[, intersect(names(data), names(grid)), drop = FALSE]

  # Get predictions with probabilities
  pred_result <- svm_predict_tree(tree, grid,
    return_probs = TRUE,
    calibrate_probs = TRUE
  )

  # Create plot data - only keep the two plotting dimensions
  plot_data <- data.frame(
    x = grid[[plot_features[1]]],
    y = grid[[plot_features[2]]],
    prediction = pred_result$predictions
  )
  names(plot_data)[1:2] <- plot_features

  # Get class levels from original data
  class_levels <- levels(factor(data[[response]]))

  # # Create color palette
  # n_classes <- length(class_levels)
  # cols <- if (n_classes == 3) {
  #   c("darkorange", "purple", "cyan4")
  # } else {
  #   scales::hue_pal()(n_classes)
  # }

  # Create the plot
  p <- ggplot(plot_data, aes(
    x = .data[[plot_features[1]]],
    y = .data[[plot_features[2]]]
  )) +
    geom_tile(aes(fill = prediction), alpha = 0.25) +
    geom_point(
      data = data,
      aes(
        x = .data[[plot_features[1]]],
        y = .data[[plot_features[2]]],
        color = .data[[response]],
        shape = .data[[response]]
      ),
      alpha = 0.5
    ) +
    scale_color_brewer(palette = "Dark2") +
    scale_fill_brewer(palette = "Dark2") +
    labs(
      x = plot_features[1],
      y = plot_features[2]
    ) +
    theme_minimal() +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(fill = NA),
      legend.position = "bottom",
      legend.text = element_text(size = 8),
      legend.title = element_blank(),
      aspect.ratio = 1
    ) +
    guides(fill = "none")

  return(p)
}

get_tree_features <- function(tree) {
  if (tree$is_leaf) {
    return(tree$features)
  }

  features <- tree$features

  if (!is.null(tree$left)) {
    features <- c(features, get_tree_features(tree$left))
  }

  if (!is.null(tree$right)) {
    features <- c(features, get_tree_features(tree$right))
  }

  return(unique(features))
}
