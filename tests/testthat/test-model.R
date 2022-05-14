test_that("can initialize and train a model", {

  dataset <- time_series_dataset(
    walmart_recipe(),
    walmart_data(),
    input_types = walmart_input_types(),
    lookback = 120,
    horizon = 4
  )

  module <- temporal_fusion_transformer(
    transform(dataset)
  )

  expect_true(inherits(module, "luz_module_generator"))
  expect_true(inherits(module, "tft_module"))

  expect_error(regexp = NA, {
    result <- module %>% fit(transform(dataset), epochs = 1, verbose  = FALSE)
  })

  expect_true(inherits(result, "luz_module_fitted"))
  expect_true(inherits(result, "tft_result"))
})

test_that("Can pass validation data to the model", {

  init <- max(walmart_data()$Date) -lubridate::weeks(8)
  train <- walmart_data() %>%
    dplyr::filter(Date <= init)
  test <- walmart_data() %>%
    dplyr::filter(Date > init) %>%
    dplyr::filter(Store == 1, Dept == 1)

  dataset <- time_series_dataset(
    walmart_recipe(),
    train,
    input_types = walmart_input_types(),
    lookback = 120,
    horizon = 4
  )

  train_ds <- transform(dataset)
  valid_ds <- transform(dataset, new_data = test)

  module <- temporal_fusion_transformer(train_ds)

  expect_error(regexp = NA, {
    result <- module %>% fit(train_ds, epochs = 1, verbose  = FALSE,
                             valid_data = valid_ds)
  })

  expect_true(inherits(result, "luz_module_fitted"))
  expect_true(any("valid" %in% luz::get_metrics(result)$set))
})

