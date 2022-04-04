step_include_roles <- function(
    recipe,
    roles = NULL,
    trained = FALSE,
    skip = FALSE,
    role = NA,
    id = recipes::rand_id("include_roles")
) {
  recipes::add_step(
    recipe,
    step_include_roles_new(
      role = role,
      roles = roles,
      trained = trained,
      skip = skip,
      id = id
    )
  )
}

step_include_roles_new <- function(role, roles, trained, skip, id) {
  recipes::step(
    subclass = "include_roles",
    role = role,
    roles = roles,
    trained = trained,
    skip = skip,
    id = id
  )
}

#' @export
prep.step_include_roles <- function(x, training, info = NULL, ...) {
  info$size <- purrr::map_int(info$variable, function(x) {
    var <- training[[x]]
    if (is.factor(var) || is.character(var))
      length(unique(var))
    else
      NA_integer_
  })

  step_include_roles_new(
    role = x$role,
    roles = info,
    trained = TRUE,
    skip = x$skip,
    id = x$id
  )
}

#' @export
bake.step_include_roles <- function(object, new_data, ...) {
  attr(new_data, "roles") <- object$roles
  new_data
}
