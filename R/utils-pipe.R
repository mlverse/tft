#' Pipe operator
#'
#' See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
NULL

#' @importFrom zeallot %<-%
NULL

#' Fit a model
#' See [generics::fit()] for more information.
#' @keywords internal
#' @rdname fit
#' @name fit
#' @importFrom generics fit
#' @export
NULL

#' Prepare a specification
#' See [recipes::prep()] for more information.
#' @export
#' @importFrom recipes prep
#' @rdname prep
#' @keywords internal
#' @name prep
NULL

#' Creates forecasts
#' See [generics::forecast()] for more information.
#' @export
#' @importFrom generics forecast
#' @name forecast
#' @keywords internal
NULL

utils::globalVariables(c(".stats", ".", ".min_index", ":=", "..mean", "..sd",
                         "type", "variable", "object", "new_data"))
