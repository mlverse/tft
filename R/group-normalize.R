#' Group normalization
#'
#' [recipes::recipe()] step for normalizing data per group.
#' Most of the times we want to normalize each time series independently as
#' they might have different scales.
#'
#' @inheritParams recipes::step_normalize
#' @param groups <[`tidy-select`][tidyr::tidyr_tidy_select]> Columns to group by
#'   before computing normalization statistics.
#' @param stats Is modified by `prep`. A data frame containing one row per distinct
#'   group, containing the normalization statistics.
#'
#' @export
step_group_normalize <- function(
    recipe,
    ...,
    groups,
    stats = NULL,
    role = NA,
    trained = FALSE,
    skip = FALSE,
    id = recipes::rand_id("group_normalize")
) {

  terms <- recipes::ellipse_check(...)
  groups <- rlang::enquos(groups)

  recipes::add_step(
    recipe,
    step_group_normalize_new(
      terms = terms,
      groups = groups,
      stats = stats,
      trained = trained,
      role = role,
      skip = skip,
      id = id
    )
  )
}

step_group_normalize_new <-
  function(terms, groups, role, trained, stats, skip, id) {
    recipes::step(
      subclass = "group_normalize",
      terms = terms,
      groups = groups,
      stats = stats,
      role = role,
      trained = trained,
      skip = skip,
      id = id
    )
  }

#' @importFrom recipes prep
#' @export
prep.step_group_normalize <- function(x, training, info = NULL, ...) {
  col_names <- recipes::recipes_eval_select(x$terms, training, info)
  groups <- recipes::recipes_eval_select(x$groups, training, info)

  stats <- training %>%
    dplyr::group_by(!!!rlang::syms(groups)) %>%
    dplyr::summarise(dplyr::across(dplyr::all_of(col_names), c(mean = mean, sd = sd))) %>%
    tidyr::pivot_longer(
      cols = c(dplyr::ends_with("_mean"), dplyr::ends_with("_sd")),
      names_to = c(".column", ".stat"),
      names_sep = "_",
      values_to = ".value"
    ) %>%
    tidyr::pivot_wider(
      names_from = ".stat",
      values_from = ".value"
    ) %>%
    dplyr::group_nest(!!!rlang::syms(groups), .key = ".stats")

  attr(stats, "groups") <- groups
  attr(stats, "columns") <- col_names

  step_group_normalize_new(
    terms = x$terms,
    groups = x$groups,
    trained = TRUE,
    role = x$role,
    stats = stats,
    skip = x$skip,
    id = x$id
  )
}

#' @importFrom recipes bake
#' @export
bake.step_group_normalize <- function(object, new_data, ...) {
  columns <- attr(object$stats, "columns")
  keys <- attr(object$stats, "groups")

  normalize <- function(col, .stats) {
    stats <- .stats[.stats$.column == dplyr::cur_column(),]
    (col - stats$mean)/stats$sd
  }

  new_data %>%
    dplyr::left_join(object$stats, by = keys) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(dplyr::across(
      .cols = dplyr::all_of(columns),
      ~normalize(.x, .stats)
    )) %>%
    dplyr::select(-.stats) %>%
    dplyr::ungroup()
}
