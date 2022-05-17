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

  predictions <- predict(result, new_data = test)
  expect_equal(nrow(predictions), 4)
  expect_equal(ncol(predictions), 3)

})

test_that("verification is working", {

  init <- max(walmart_data()$Date) - lubridate::weeks(8)
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

  expect_error(
    predict(result, new_data = test),
    regexp = "includes obs that we can't generate predictions"
  )

  expect_error(
    predict(result, new_data = test, past_data = test),
    regexp = "includes obs that we can't"
  )

  test2 <- test
  test2$Size[1] <- NA

  expect_error(
    predict(result, new_data = test2),
    regexp = "Found missing values in at"
  )

  test2 <- test[1:3,]
  expect_error(
    predict(result, new_data = test2),
    regexp = "At least one group"
  )

})

