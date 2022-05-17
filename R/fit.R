
# by default we normalize the outcomes per group.
#' @importFrom stats sd
#' @importFrom utils tail
normalize_outcome <- function(x, keys, outcome, constants = NULL) {
  outcome <- rlang::sym(outcome)

  if (is.null(constants)) {
    constants <- x %>%
      tibble::as_tibble() %>%
      dplyr::ungroup() %>%
      dplyr::group_by(!!!rlang::syms(keys)) %>%
      dplyr::summarise(.groups = "drop",
                       ..mean := mean({{outcome}}),
                       ..sd := sd({{outcome}})
      )
  }

  x <- x %>%
    dplyr::left_join(constants, by = keys) %>%
    dplyr::mutate({{outcome}} := ({{outcome}} - ..mean)/..sd) %>%
    dplyr::select(-..mean, -..sd)

  list(constants = constants, x = x)
}

unnormalize_outcome <- function(x, constants, outcome) {
  keys <- names(constants)
  keys <- keys[!keys %in% c("..mean", "..sd")]

  outcome <- rlang::sym(outcome)

  x %>%
    dplyr::left_join(constants, by = keys) %>%
    dplyr::mutate({{outcome}} := {{outcome}} *..sd + ..mean) %>%
    dplyr::select(-..mean, -..sd)
}

model_to_raw <- function(model) {
  con <- rawConnection(raw(), open = "wr")
  luz::luz_save(model, con)
  on.exit({close(con)}, add = TRUE)
  r <- rawConnectionValue(con)
  r
}

is_null_external_pointer <- function(pointer) {
  a <- attributes(pointer)
  attributes(pointer) <- NULL
  out <- identical(pointer, methods::new("externalptr"))
  attributes(pointer) <- a
  out
}

reload_model <- function(object) {
  con <- rawConnection(object)
  on.exit({close(con)}, add = TRUE)
  module <- luz::luz_load(con)
  module
}

covariates_spec <- function(index, keys, static = NULL, known = NULL, unknown = NULL) {
  make_input_types(
    index = {{index}},
    keys = {{keys}},
    static = {{static}},
    known = {{known}},
    unknown = {{unknown}}
  )
}

make_input_types <- function(index, keys, static = NULL, known = NULL,
                             unknown = NULL) {
  output <- list(
    index = rlang::enexpr(index),
    keys = rlang::enexpr(keys),
    static = rlang::enexpr(static),
    known = rlang::enexpr(known),
    unknown = rlang::enexpr(unknown)
  )
  output
}

evaluate_types <- function(data, types) {
  types <- lapply(types, function(x) {
    colnames(dplyr::select(data, !!!unlist(x)))
  })
  # Non-specified variables are considered unknown.
  unknown <- names(data)[!names(data) %in% unlist(types)]
  types[["unknown"]] <- c(types[["unknown"]], unknown)
  types
}
