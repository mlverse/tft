input_types <- list(
  target = "outcome",
  observed_input = "observerd_input",
  known_input = "known_input",
  static_input = "static_input",
  id = "id",
  time = "time"
)

role_observed <- function(recipe, ...) {
  recipes::add_role(recipe, ..., new_role = input_types$observed_input)
}

all_observed <- function() {
  recipes::has_role(input_types$observed_input)
}

role_known <-function(recipe, ...) {
  recipes::add_role(recipe, ..., new_role = input_types$known_input)
}

all_known <- function() {
  recipes::has_role(input_types$known_input)
}

role_static <- function(recipe, ...) {
  recipes::add_role(recipe, ..., new_role = input_types$static_input)
}

all_static <- function() {
  recipes::has_role(input_types$static_input)
}

role_id <- function(recipe, ...) {
  recipes::add_role(recipe, ..., new_role = input_types$id)
}

all_id <- function() {
  recipes::has_role("id")
}

role_time <- function(recipe, ...) {
  recipes::add_role(recipe, ..., new_role = input_types$time)
}

all_time <- function() {
  recipes::has_role(input_types$time)
}
