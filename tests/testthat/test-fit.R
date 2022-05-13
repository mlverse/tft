test_that("first fit", {
  expect_error(
    result <- tft(walmart_recipe(), walmart_data(), lookback = 120, horizon = 4,
                  epochs = 1, input_types = walmart_input_types()),
    regexp = NA
  )
})

test_that("can pass validation data to fit", {

  init <- max(walmart_data()$Date) -lubridate::weeks(8)
  train <- walmart_data() %>%
    dplyr::filter(Date <= init)
  test <- walmart_data() %>%
    dplyr::filter(Date > init) %>%
    dplyr::filter(Store == 1, Dept == 1)

  set.seed(1)
  torch::torch_manual_seed(1)
  expect_error({
    result1 <- tft(walmart_recipe(), train, lookback = 120, horizon = 4,
                  epochs = 1, input_types = walmart_input_types(),
                  valid_data = test)
  }, regexp = NA)

  set.seed(1)
  torch::torch_manual_seed(1)
  expect_error({
    result2 <- tft(walmart_recipe(), train, lookback = 120, horizon = 4,
                  epochs = 1, input_types = walmart_input_types(),
                  valid_data = list(train, test))
  }, regexp = NA)

  expect_identical(
    luz::get_metrics(result1$module),
    luz::get_metrics(result2$module)
  )
})
