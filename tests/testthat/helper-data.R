
walmart_data <- function() {
  df <- walmartdata::walmart_sales %>%
    dplyr::mutate(
      Store = as.character(Store),
      Dept = as.character(Dept)
    ) %>%
    dplyr::filter(Store %in% c(1, 2), Dept %in% c(1,2))%>%
    tsibble::tsibble(
      key = c(Store, Dept, Type, Size),
      index = Date
    ) %>%
    tsibble::group_by_key() %>%
    tsibble::fill_gaps(
      Weekly_Sales = 0,
      IsHoliday = FALSE
    ) %>%
    tidyr::fill(Size, Temperature, Fuel_Price, CPI, Unemployment, .direction = "down")
  df
}

walmart_recipe <- function(df = walmart_data()) {
  recipe <- recipes::recipe(Weekly_Sales ~ ., data = df) %>%
    recipes::step_date(Date, role = "known", features = c("year", "month", "doy")) %>%
    recipes::step_normalize(recipes::all_numeric_predictors()) %>%
    recipes::step_indicate_na(dplyr::starts_with("MarkDown")) %>%
    recipes::step_impute_mean(dplyr::starts_with("Markdown")) %>%
    recipes::step_mutate(
      intercept = 1
    )
  recipe
}

walmart_input_types <- function() {
  make_input_types(
    index = Date,
    keys = c(Store, Dept),
    static = c(Type, Size),
    known = c(starts_with("MarkDown"), starts_with("Date_"), starts_with("na_ind"))
  )
}

walmart_input_types_no_known <- function() {
  make_input_types(
    index = Date,
    keys = c(Store, Dept),
    static = c(),
    known = c(starts_with("Date_"), intercept)
  )
}
