
walmart_data <- function() {
  df <- timetk::walmart_sales_weekly %>%
    dplyr::select(-id) %>%
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

  df <- dplyr::bind_rows(df, df %>% dplyr::mutate(Store = "2"))
  df <- dplyr::bind_rows(df, df %>% dplyr::mutate(Dept = "2"))
  df$Size <- df$Size + as.numeric(df$Store)

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

walmart_spec <- function(recipe = walmart_recipe(), data = walmart_data()) {
  spec <- tft_dataset_spec(recipe, data) %>%
    spec_time_splits(lookback = 52, horizon = 4) %>%
    spec_covariate_index(Date) %>%
    spec_covariate_key(Store, Dept) %>%
    spec_covariate_static(Type, Size) %>%
    spec_covariate_known(starts_with("MarkDown"), starts_with("Date_"),
                         starts_with("na_ind")) %>%
    prep()
  spec
}
