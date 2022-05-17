test_that("can initialize and train a model", {

  spec <- walmart_spec()

  module <- temporal_fusion_transformer(spec)

  expect_true(inherits(module, "luz_module_generator"))
  expect_true(inherits(module, "tft_module"))

  expect_error(regexp = NA, {
    result <- module %>% fit(transform(spec), epochs = 1, verbose  = FALSE)
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

  spec <- walmart_spec(data = train)

  train_ds <- transform(spec)
  valid_ds <- transform(spec, new_data = test)

  module <- temporal_fusion_transformer(spec)

  expect_error(regexp = NA, {
    result <- module %>% fit(train_ds, epochs = 1, verbose  = FALSE,
                             valid_data = valid_ds)
  })

  expect_true(inherits(result, "luz_module_fitted"))
  expect_true(any("valid" %in% luz::get_metrics(result)$set))
})

test_that("can make predictions", {
  init <- max(walmart_data()$Date) -lubridate::weeks(4)
  train <- walmart_data() %>%
    dplyr::filter(Date <= init)
  test <- walmart_data() %>%
    dplyr::filter(Date > init) %>%
    dplyr::filter(Store == 1, Dept == 1)

  spec <- walmart_spec(data = train)

  train_ds <- transform(spec)
  valid_ds <- transform(spec, new_data = test)

  module <- temporal_fusion_transformer(spec)
  result <- module %>% fit(train_ds, epochs = 1, verbose  = FALSE,
                           valid_data = valid_ds)

  predict(result, new_data = test)
})

