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

#' @importFrom generics fit
NULL

utils::globalVariables(c(".stats", ".", ".min_index", ":=", "..mean", "..sd",
                         "type", "variable", "object", "new_data"))
