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

test_that("rolling predict works", {

  init <- max(walmart_data()$Date) - lubridate::weeks(8)
  train <- walmart_data() %>%
    dplyr::filter(Date <= init)
  test <- walmart_data() %>%
    dplyr::filter(Date > init) %>%
    dplyr::filter(Store == 1, Dept == 1)

  spec <- walmart_spec(data = train)

  train_ds <- transform(spec, subsample = 0.1)
  valid_ds <- transform(spec, new_data = test)

  module <- temporal_fusion_transformer(spec)
  result <- module %>% fit(train_ds, epochs = 1, verbose  = FALSE,
                           valid_data = valid_ds)

  predictions <- rolling_predict(result, new_data = test, past_data = train)

  expect_equal(nrow(predictions), 2)
  expect_equal(ncol(predictions), 3)
  for (p in predictions$.pred) {
    expect_equal(nrow(p), 4)
    expect_equal(ncol(p), 3)
  }

})

test_that("forecast works", {

  init <- max(walmart_data()$Date) - lubridate::weeks(8)
  train <- walmart_data() %>%
    dplyr::filter(Date <= init)
  test <- walmart_data() %>%
    dplyr::filter(Date > init) %>%
    dplyr::filter(Store == 1, Dept == 1)

  spec <- tft_dataset_spec(walmart_recipe(), train) %>%
    spec_time_splits(lookback = 52, horizon = 4) %>%
    spec_covariate_index(Date) %>%
    spec_covariate_key(Store, Dept, Type) %>%
    spec_covariate_known(intercept) %>%
    prep()

  train_ds <- transform(spec)

  model <- temporal_fusion_transformer(spec)

  result <- fit(model, train_ds, epochs = 1, verbose = FALSE)

  preds <- forecast(result)

  expect_s3_class(preds, "tft_forecast")
  expect_equal(nrow(preds), 16)
  expect_equal(ncol(preds), 7)

})

test_that("serialization works", {
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

  pred1 <- predict(result, new_data = test)

  tmp <- tempfile()
  saveRDS(result, tmp)

  rm(result); gc(); gc();

  result <- readRDS(tmp)
  pred2 <- predict(result, new_data = test)

  expect_equal(pred1, pred2)
  expect_equal(as.numeric(result$model$.check), 1)

})

