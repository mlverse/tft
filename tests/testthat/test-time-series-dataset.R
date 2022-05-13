test_that("simple test for time series dataset", {

  sales <- walmartdata::walmart_sales %>%
    dplyr::mutate(
      Store = as.character(Store),
      Dept = as.character(Dept),
      Type = as.character(Type)
    ) %>%
    dplyr::filter(Store %in% c(1, 2), Dept %in% c(1,2)) %>%
    tsibble::tsibble(
      key = c(Store, Dept, Type, Size),
      index = Date
    ) %>%
    tsibble::group_by_key() %>%
    tsibble::fill_gaps(
      Weekly_Sales = 0,
      IsHoliday = FALSE
    ) %>%
    tidyr::fill(Size, Temperature, Fuel_Price, CPI, Unemployment, .direction = "down") %>%
    dplyr::select(-starts_with("MarkDown"), -IsHoliday)

  recipe <- recipes::recipe(Weekly_Sales ~ ., data = sales) %>%
    recipes::prep()

  dataset <- time_series_dataset_generator(
    recipes::juice(recipe),
    evaluate_types(recipes::juice(recipe) %>% tibble::as_tibble(), make_input_types(
      index = Date,
      keys = c(Store, Dept),
      static = c(Type, Size)
    )),
    lookback = 6, assess_stop = 4
  )

  counts <- sales %>%
    tibble::as_tibble() %>%
    dplyr::ungroup() %>%
    dplyr::group_by(Store, Dept, Type, Size) %>%
    dplyr::count() %>%
    dplyr::mutate(
      n = n - (6 + 4 - 1),
      n = ifelse(n > 0, n, 0)
    )

  expect_equal(length(dataset), sum(counts$n))
  expect_error(obs <- dataset[1], regex = NA)

})

test_that("Good error message when no group has enough lookback", {

  recipe <- walmart_recipe() %>%
    recipes::prep()

  expect_error(regexp = "No group", {
    dataset <- time_series_dataset_generator(
      recipes::juice(recipe),
      evaluate_types(
        recipes::juice(recipe) %>% tibble::as_tibble(),
        walmart_input_types()
      ),
      lookback = 500, assess_stop = 4)
  })

})

test_that("Time series dataset creators", {

  dataset <- time_series_dataset(
    walmart_recipe(),
    walmart_data(),
    input_types = walmart_input_types(),
    lookback = 120,
    horizon = 4
  )

  expect_true(inherits(dataset, "time_series_dataset"))
  expect_true(inherits(dataset, "hardhat_model"))

  dt <- transform(dataset)

  expect_true(inherits(dt, "time_series_dataset"))
  expect_true(inherits(dt, "dataset"))
})

