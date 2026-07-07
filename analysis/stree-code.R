select_best_ovr_split <- function(impurities_list, tie_break = "first") {
  # Extract impurities
  impurities <- sapply(impurities_list, function(x) {
    if (!is.null(x$impurity) && !is.na(x$impurity)) x$impurity else Inf
  })

  # Find minimum
  min_impurity <- min(impurities[is.finite(impurities)])

  if (!is.finite(min_impurity)) {
    return(NULL)
  }

  # Find all classes with minimum impurity (for tie-breaking)
  best_classes <- names(impurities)[impurities == min_impurity]

  if (length(best_classes) == 1) {
    return(best_classes[1])
  }

  # Tie-breaking
  switch(tie_break,
    "first" = best_classes[1], # Choose first in order
    "last" = best_classes[length(best_classes)],
    "random" = sample(best_classes, 1),
    best_classes[1] # Default to first
  )
}


stree_fit_binary_svm <- function(X, y, kernel, verbose = FALSE,
                                 use_scaling = TRUE, scaling_params = NULL, ...) {
  if (nrow(X) == 0 || length(y) == 0) {
    if (verbose) cat("Empty data, cannot fit SVM\n")
    return(list(
      model = NULL, left_idx = NULL, right_idx = NULL,
      scaling_params = NULL, used_features = NULL
    ))
  }

  # Convert factors to numeric
  X_processed <- X
  for (col in names(X_processed)) {
    if (is.factor(X_processed[[col]])) {
      X_processed[[col]] <- as.numeric(X_processed[[col]])
    } else if (is.character(X_processed[[col]])) {
      X_processed[[col]] <- as.numeric(as.factor(X_processed[[col]]))
    }
  }

  # Remove constant columns (no variance)
  col_vars <- sapply(X_processed, function(col) {
    if (is.numeric(col)) var(col, na.rm = TRUE) else 1
  })
  non_constant_cols <- col_vars > 1e-10

  if (sum(non_constant_cols) == 0) {
    if (verbose) cat("All features are constant, cannot fit SVM\n")
    return(list(
      model = NULL, left_idx = NULL, right_idx = NULL,
      scaling_params = NULL, used_features = NULL
    ))
  }

  X_filtered <- X_processed[, non_constant_cols, drop = FALSE]

  # Apply custom scaling (matches sklearn)
  if (use_scaling) {
    if (is.null(scaling_params)) {
      scaled_result <- standard_scaler(X_filtered,
        center = TRUE,
        scale_var = TRUE, return_params = TRUE
      )
      X_scaled <- scaled_result$data
      scaling_params <- scaled_result$params
    } else {
      X_scaled <- standard_scaler(X_filtered, params = scaling_params)
    }
  } else {
    X_scaled <- X_filtered
    scaling_params <- NULL
  }

  # Map kernel names
  svm_kernel <- switch(kernel,
    "linear" = "linear",
    "polynomial" = "polynomial",
    "radial" = "radial",
    "linear"
  )

  # Fit SVM
  model <- tryCatch(
    {
      e1071::svm(
        x = X_scaled,
        y = y,
        kernel = svm_kernel,
        scale = FALSE,
        decision.values = TRUE,
        tolerance = 0.0001,
        ...
      )
    },
    error = function(e) {
      if (verbose) cat("SVM fitting failed:", e$message, "\n")
      NULL
    }
  )

  if (is.null(model)) {
    return(list(
      model = NULL, left_idx = NULL, right_idx = NULL,
      scaling_params = NULL, used_features = NULL
    ))
  }

  # Get decision values
  dec_values <- attr(
    predict(model, X_scaled, decision.values = TRUE),
    "decision.values"
  )

  if (is.matrix(dec_values)) {
    dec_values <- dec_values[, 1]
  } else {
    dec_values <- as.numeric(dec_values)
  }

  left_idx <- which(dec_values > 0)
  right_idx <- which(dec_values <= 0)

  return(list(
    model = model,
    left_idx = left_idx,
    right_idx = right_idx,
    scaling_params = scaling_params,
    used_features = names(X_filtered)
  ))
}


get_consistent_class_order <- function(y, sort_method = "natural") {
  classes <- unique(as.character(y))

  switch(sort_method,
    "natural" = sort(classes), # Alphabetical/numerical sort
    "frequency" = {
      # Sort by frequency (most common first)
      freq_table <- table(y)
      names(sort(freq_table, decreasing = TRUE))
    },
    "original" = classes, # Order of first appearance
    sort(classes) # Default to natural sort
  )
}

stree_leaf_node <- function(y, n, all_classes) {
  # Compute class probabilities
  class_table <- table(factor(y, levels = all_classes))
  class_probs <- prop.table(class_table)

  # Handle edge case: empty probabilities
  if (sum(class_probs) == 0) {
    class_probs <- rep(1 / length(all_classes), length(all_classes))
    names(class_probs) <- all_classes
  }

  prediction <- names(which.max(class_probs))

  list(
    is_leaf = TRUE,
    prediction = prediction,
    class_prob = as.numeric(class_probs),
    class_names = all_classes,
    n = n
  )
}


stree_predict <- function(tree, newdata, return_probs = FALSE) {
  # Batch prediction
  if (is.data.frame(newdata) && nrow(newdata) > 1) {
    preds <- lapply(seq_len(nrow(newdata)), function(i) {
      stree_predict(tree, newdata[i, , drop = FALSE], return_probs)
    })

    if (return_probs) {
      pred_classes <- sapply(preds, function(x) x$prediction)
      prob_matrix <- do.call(rbind, lapply(preds, function(x) x$probabilities))
      return(list(predictions = pred_classes, probabilities = prob_matrix))
    } else {
      return(unlist(preds))
    }
  }

  # Single observation or leaf node
  if (tree$is_leaf) {
    if (return_probs) {
      prob_vector <- tree$class_prob
      names(prob_vector) <- tree$class_names
      return(list(
        prediction = tree$prediction,
        probabilities = prob_vector
      ))
    } else {
      return(tree$prediction)
    }
  }

  # Internal node: prepare data
  # *** CRITICAL: Only select features that were used in training ***
  newdata_subset <- newdata[, tree$features, drop = FALSE]

  # *** CRITICAL: Apply same preprocessing as training ***
  # Convert factors to numeric (same as training)
  for (col in names(newdata_subset)) {
    if (is.factor(newdata_subset[[col]])) {
      newdata_subset[[col]] <- as.numeric(newdata_subset[[col]])
    } else if (is.character(newdata_subset[[col]])) {
      newdata_subset[[col]] <- as.numeric(as.factor(newdata_subset[[col]]))
    }
  }

  # *** CRITICAL FIX: Apply the EXACT same scaling used during training ***
  if (!is.null(tree$scaling_params)) {
    # Ensure we're scaling exactly the same features
    newdata_subset <- standard_scaler(newdata_subset, params = tree$scaling_params)
  }

  # Get decision value from SVM
  dec_value <- tryCatch(
    {
      pred_result <- predict(tree$model, newdata_subset, decision.values = TRUE)
      attr(pred_result, "decision.values")
    },
    error = function(e) {
      warning("Prediction failed: ", e$message)
      return(0) # Default to 0 if prediction fails
    }
  )

  # Extract scalar value
  if (is.matrix(dec_value)) {
    dec_value <- dec_value[1, 1]
  } else {
    dec_value <- as.numeric(dec_value)[1]
  }

  # *** CRITICAL FIX: Use > instead of >= to match Python (line 772) ***
  # Python: self._up = data > 0
  if (dec_value > 0) { # Changed from >= to >
    return(stree_predict(tree$left, newdata, return_probs))
  } else {
    return(stree_predict(tree$right, newdata, return_probs))
  }
}

print_stree <- function(tree, indent = "", show_probs = FALSE) {
  if (tree$is_leaf) {
    cat(indent, "[Leaf] predict =", tree$prediction, "| n =", tree$n)
    if (show_probs) {
      cat(" | probs = [", paste(round(tree$class_prob, 3), collapse = ", "), "]")
    }
    cat("\n")
    return(invisible())
  }

  cat(indent, "[Node] depth =", tree$depth, "| n =", tree$n, "| kernel =", tree$kernel)
  if (!is.null(tree$hyperplane_class)) {
    cat(" | split:", tree$hyperplane_class, "vs rest")
  }
  if (!is.null(tree$impurity)) {
    cat(" | impurity =", round(tree$impurity, 4))
  }
  cat("\n")

  cat(indent, "|- Positive branch (distance >= 0):\n")
  if (!is.null(tree$left)) {
    print_stree(tree$left, paste0(indent, "|  "), show_probs)
  }

  cat(indent, "`- Negative branch (distance < 0):\n")
  if (!is.null(tree$right)) {
    print_stree(tree$right, paste0(indent, "   "), show_probs)
  }

  invisible()
}

standard_scaler <- function(X, center = TRUE, scale_var = TRUE,
                            return_params = FALSE, params = NULL) {
  # Convert to data frame if matrix
  if (is.matrix(X)) {
    X <- as.data.frame(X)
  }

  # Identify numeric columns
  numeric_cols <- sapply(X, is.numeric)

  if (sum(numeric_cols) == 0) {
    warning("No numeric columns to scale")
    if (return_params) {
      return(list(data = X, params = list(means = NULL, sds = NULL)))
    } else {
      return(X)
    }
  }

  X_scaled <- X

  # If params provided (for test data), use those
  if (!is.null(params)) {
    for (col in names(params$means)) {
      if (col %in% names(X_scaled)) {
        if (center) {
          X_scaled[[col]] <- X_scaled[[col]] - params$means[[col]]
        }
        if (scale_var && params$sds[[col]] > 0) {
          X_scaled[[col]] <- X_scaled[[col]] / params$sds[[col]]
        }
      }
    }
    return(X_scaled)
  }

  # Otherwise, compute parameters from data
  means <- list()
  sds <- list()

  for (col in names(X_scaled)[numeric_cols]) {
    col_data <- X_scaled[[col]]

    # Remove NA for calculation
    col_data_clean <- col_data[!is.na(col_data)]

    if (length(col_data_clean) == 0) {
      means[[col]] <- 0
      sds[[col]] <- 1
      next
    }

    # Calculate mean
    col_mean <- if (center) mean(col_data_clean, na.rm = TRUE) else 0
    means[[col]] <- col_mean

    # Calculate standard deviation (using n, not n-1 like R's default)
    # This matches sklearn's behavior
    if (scale_var) {
      if (length(col_data_clean) > 1) {
        # sklearn uses population std (divides by n, not n-1)
        col_sd <- sqrt(sum((col_data_clean - col_mean)^2) / length(col_data_clean))
      } else {
        col_sd <- 1
      }

      # Avoid division by zero
      if (col_sd == 0 || is.na(col_sd)) {
        col_sd <- 1
      }
    } else {
      col_sd <- 1
    }
    sds[[col]] <- col_sd

    # Apply transformation
    if (center) {
      X_scaled[[col]] <- X_scaled[[col]] - col_mean
    }
    if (scale_var) {
      X_scaled[[col]] <- X_scaled[[col]] / col_sd
    }
  }

  if (return_params) {
    return(list(
      data = X_scaled,
      params = list(means = means, sds = sds)
    ))
  } else {
    return(X_scaled)
  }
}


stree_split <- function(data, response, depth = 1, max_depth = 5,
                        min_samples = 5, kernel = "linear",
                        impurity_measure = "entropy",
                        split_criteria = "impurity", # Added parameter
                        cost = 1, verbose = FALSE,
                        all_classes = NULL,
                        class_order_method = "natural",
                        tie_break_method = "first", ...) {
  # Initialize all_classes if NULL
  if (is.null(all_classes)) {
    all_classes <- get_consistent_class_order(data[[response]], class_order_method)
  }

  if (verbose) {
    cat("\n--- STree Node at depth", depth, "---\n")
    cat("Samples:", nrow(data), "\n")
    cat("Class distribution:\n")
    print(table(data[[response]]))
  }

  # Handle NA rows
  if (anyNA(data)) {
    if (verbose) cat("Warning: NA values detected! Stopping here.\n")
    return(leaf_node(data[[response]], nrow(data), all_classes))
  }

  y <- data[[response]]
  n <- nrow(data)

  # Stopping conditions
  if (depth > max_depth || length(unique(y)) == 1 || n < min_samples) {
    if (verbose) cat("Creating leaf node\n")
    return(leaf_node(y, n, all_classes))
  }

  # Get unique classes at this node
  present_classes <- intersect(all_classes, unique(as.character(y)))
  k <- length(present_classes)

  if (verbose) cat("Number of classes at node:", k, "\n")

  # Prepare features
  features <- setdiff(names(data), response)
  X <- data[features]

  # Binary case - same as before
  if (k == 2) {
    if (verbose) cat("Binary classification case\n")

    result <- stree_fit_binary_svm( # Use fixed version
      X, factor(y), kernel,
      verbose = verbose,
      use_scaling = TRUE, cost = cost, ...
    )

    if (is.null(result$model)) {
      return(leaf_node(y, n, all_classes))
    }

    left_idx <- result$left_idx
    right_idx <- result$right_idx

    if (length(left_idx) == 0 || length(right_idx) == 0 ||
      length(left_idx) < min_samples || length(right_idx) < min_samples) {
      return(leaf_node(y, n, all_classes))
    }

    # Recursive calls
    left_child <- stree_split(
      data[left_idx, ], response, depth + 1, max_depth, min_samples,
      kernel, impurity_measure, split_criteria, cost, verbose, all_classes,
      class_order_method, tie_break_method, ...
    )

    right_child <- stree_split(
      data[right_idx, ], response, depth + 1, max_depth, min_samples,
      kernel, impurity_measure, split_criteria, cost, verbose, all_classes,
      class_order_method, tie_break_method, ...
    )

    return(list(
      is_leaf = FALSE,
      model = result$model,
      features = result$used_features,
      scaling_params = result$scaling_params,
      hyperplane_class = NULL,
      partition_column = 1, # Store for prediction
      left = left_child,
      right = right_child,
      depth = depth,
      n = n,
      kernel = kernel
    ))
  }

  # Multiclass case with improved OVR
  if (verbose) cat("Multi-class case: trying", k, "one-vs-rest splits\n")

  impurity_func <- if (impurity_measure == "entropy") entropy else gini
  impurities_list <- list()

  for (target_class in present_classes) {
    y_binary <- factor(
      ifelse(y == target_class, "positive", "negative"),
      levels = c("positive", "negative")
    )

    if (verbose) cat("  Trying:", target_class, "vs rest\n")

    result <- stree_fit_binary_svm( # Use fixed version
      X, y_binary, kernel,
      verbose = FALSE,
      use_scaling = TRUE, cost = cost, ...
    )

    if (is.null(result$model)) {
      impurities_list[[target_class]] <- list(impurity = NA)
      next
    }

    left_idx <- result$left_idx
    right_idx <- result$right_idx

    if (length(left_idx) == 0 || length(right_idx) == 0) {
      impurities_list[[target_class]] <- list(impurity = NA)
      next
    }

    # Calculate weighted impurity
    y_left <- y[left_idx]
    y_right <- y[right_idx]

    impurity_left <- impurity_func(y_left)
    impurity_right <- impurity_func(y_right)

    weighted_impurity <- (length(left_idx) / n) * impurity_left +
      (length(right_idx) / n) * impurity_right

    if (verbose) {
      cat("    Weighted impurity:", round(weighted_impurity, 4), "\n")
    }

    impurities_list[[target_class]] <- list(
      impurity = weighted_impurity,
      model = result$model,
      left_idx = left_idx,
      right_idx = right_idx,
      scaling_params = result$scaling_params,
      used_features = result$used_features
    )
  }

  # Select best split
  best_class <- select_best_ovr_split(impurities_list, tie_break_method)

  if (is.null(best_class)) {
    if (verbose) cat("No valid split found, creating leaf\n")
    return(leaf_node(y, n, all_classes))
  }

  best_result <- impurities_list[[best_class]]

  if (verbose) {
    cat("Best split: class", best_class, "vs rest\n")
    cat("Best impurity:", round(best_result$impurity, 4), "\n")
  }

  # Check child sizes
  if (length(best_result$left_idx) < min_samples ||
    length(best_result$right_idx) < min_samples) {
    return(leaf_node(y, n, all_classes))
  }

  # Store which class was selected for this split
  partition_column <- which(present_classes == best_class)

  # Recursive calls
  left_child <- stree_split(
    data[best_result$left_idx, ], response, depth + 1, max_depth, min_samples,
    kernel, impurity_measure, split_criteria, cost, verbose, all_classes,
    class_order_method, tie_break_method, ...
  )

  right_child <- stree_split(
    data[best_result$right_idx, ], response, depth + 1, max_depth, min_samples,
    kernel, impurity_measure, split_criteria, cost, verbose, all_classes,
    class_order_method, tie_break_method, ...
  )

  return(list(
    is_leaf = FALSE,
    model = best_result$model,
    features = best_result$used_features,
    scaling_params = best_result$scaling_params,
    hyperplane_class = best_class,
    partition_column = partition_column, # Store for prediction
    left = left_child,
    right = right_child,
    depth = depth,
    n = n,
    impurity = best_result$impurity,
    kernel = kernel
  ))
}
