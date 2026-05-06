# Navigate tree and extract feature usage
get_tree_feature_usage <- function(tree, depth = 0, feature_log = list()) {
  # Base case: leaf node
  if (isTRUE(tree$is_leaf)) {
    return(feature_log)
  }

  # Record features used at this node
  if (!is.null(tree$features)) {
    for (feat in tree$features) {
      if (is.null(feature_log[[feat]])) {
        feature_log[[feat]] <- list(count = 0, depths = c())
      }
      feature_log[[feat]]$count <- feature_log[[feat]]$count + 1
      feature_log[[feat]]$depths <- c(feature_log[[feat]]$depths, depth)
    }
  }

  # Recurse into children
  if (!is.null(tree$left)) {
    feature_log <- get_tree_feature_usage(tree$left, depth + 1, feature_log)
  }
  if (!is.null(tree$right)) {
    feature_log <- get_tree_feature_usage(tree$right, depth + 1, feature_log)
  }

  return(feature_log)
}

# Summarise into tidy tibble
summarise_feature_usage <- function(tree, top_n = NULL) {
  usage <- get_tree_feature_usage(tree)

  if (length(usage) == 0) {
    cat("No internal nodes found — tree may be a single leaf.\n")
    return(tibble())
  }

  tbl <- map_df(names(usage), function(feat) {
    tibble(
      feature    = feat,
      count      = usage[[feat]]$count,
      mean_depth = round(mean(usage[[feat]]$depths), 2),
      min_depth  = min(usage[[feat]]$depths),
      max_depth  = max(usage[[feat]]$depths),
      depths     = list(usage[[feat]]$depths)
    )
  }) |>
    arrange(desc(count))

  if (!is.null(top_n)) tbl <- slice(tbl, 1:min(top_n, nrow(tbl)))

  tbl
}
