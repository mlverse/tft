
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

walmart_recipe <- function() {
  recipe <- recipes::recipe(Weekly_Sales ~ ., data = df) %>%
    recipes::update_role(!!!tsibble::key_vars(df), new_role = "key") %>%
    recipes::update_role(!!!tsibble::index_var(df), new_role = "index") %>%
    recipes::step_date(Date, role = "known", features = c("year", "month", "doy")) %>%
    recipes::update_role(IsHoliday, new_role = "unused") %>%
    recipes::step_normalize(all_numeric_predictors()) %>%
    recipes::step_indicate_na(starts_with("MarkDown")) %>%
    recipes::step_impute_mean(starts_with("Markdown")) %>%
    step_include_roles()
  recipe
}
