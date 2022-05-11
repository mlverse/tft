test_that("future_data works in a simple case", {

  x <- data.frame(
    k = "hello",
    x = lubridate::ymd("2010-01-01") + lubridate::days(1:15),
    y = runif(15)
  )

  x <- tsibble::as_tsibble(x, key = k, index = x)
  expect_equal(nrow(future_data(x, 3)), 3)

  x <- data.frame(
    k = c(rep("hello", 15), rep("bye", 15)),
    x = c(
      lubridate::ymd("2010-01-01") + lubridate::days(1:15),
      lubridate::ymd("2010-01-01") + lubridate::days(1:15)
    ),
    y = runif(15)
  )
  x <- tsibble::as_tsibble(x, key = k, index = x)

  expect_equal(nrow(future_data(x, 3)), 6)
})

test_that("future data correcty get's rid of obs that we can't make predictions", {

  # in this case, since `bye` is not complete until the end. ie, until '2010-01-16'
  # we can't make predictions for it.
  x <- data.frame(
    k = c(rep("hello", 15), rep("bye", 15)),
    x = c(
      lubridate::ymd("2010-01-01") + lubridate::days(1:15),
      lubridate::ymd("2010-01-01") + lubridate::days(0:14)
    ),
    y = runif(15)
  )
  x <- tsibble::as_tsibble(x, key = k, index = x)

  expect_equal(nrow(future_data(x, 3)), 3)
})

test_that("can predict", {

  result <- tft(walmart_recipe(), walmart_data(), lookback = 120, horizon = 4,
                epochs = 1, subsample = 0.1, input_types = walmart_input_types())

  new_data <- future_data(result$past_data, 4, input_types = result$config$input_types)

  expect_error(pred <- predict(result, new_data))

  new_data <- new_data %>%
    tibble::as_tibble() %>%
    dplyr::left_join(dplyr::distinct(walmart_data(), Store, Dept, Type, Size),
                     by = c("Store", "Dept")) %>%
    dplyr::left_join(
      walmart_data() %>%
        tibble::as_tibble() %>%
        dplyr::ungroup() %>%
        dplyr::select(Date, Store, Dept, starts_with("MarkDown")) ,
      by = c("Date", "Store", "Dept")
    ) %>%
    dplyr::mutate(IsHoliday = FALSE)

  pred <- predict(result, new_data)

  expect_equal(nrow(new_data), nrow(pred))

  # expect that prediction is correctly reordered
  new_data2 <- dplyr::arrange(new_data, Dept)
  pred2 <- predict(result, new_data2)
  out <- dplyr::bind_cols(new_data, pred) %>%
    dplyr::left_join(dplyr::bind_cols(new_data2, pred2), by = names(new_data))

  expect_equal(out$.pred.x, out$.pred.y)
  expect_equal(out$.pred_lower.x, out$.pred_lower.y)
  expect_equal(out$.pred_upper.x, out$.pred_upper.y)

  # can we predict with data.frames of different size?
  new_data3 <- new_data[1:3,]
  expect_error(
    pred3 <- predict(result, new_data3),
    regexp = "At least one group"
  )

})

test_that("forecast works", {

  d <- walmart_data() %>%
    tibble::as_tibble() %>%
    dplyr::select(-IsHoliday, -Type, -Size)
  result <- tft(walmart_recipe(d), d,
                lookback = 120, horizon = 4, epochs = 1,
                input_types = walmart_input_types_no_known())

  preds <- forecast(result)
  expect_s3_class(preds, "tft_forecast")
  expect_equal(nrow(preds), 16)
  expect_equal(ncol(preds), 6)

})

test_that("full prediction, passing only future data", {

  init <- max(walmart_data()$Date) -lubridate::weeks(4)

  train <- walmart_data() %>%
    dplyr::filter(Date <= init)
  test <- walmart_data() %>%
    dplyr::filter(Date > init) %>%
    dplyr::filter(Store == 1, Dept == 1)

  result <- tft(walmart_recipe(), train, lookback = 120, horizon = 4,
                epochs = 1, input_types = walmart_input_types())

  pred <- predict(
    result,
    new_data = test
  )

  expect_equal(nrow(pred), 4)
})

test_that("can serialize and reload a model", {

  init <- max(walmart_data()$Date) -lubridate::weeks(4)
  train <- walmart_data() %>%
    dplyr::filter(Date <= init)
  test <- walmart_data() %>%
    dplyr::filter(Date > init) %>%
    dplyr::filter(Store == 1, Dept == 1)

  result <- tft(walmart_recipe(), train, lookback = 120, horizon = 4,
                epochs = 1, input_types = walmart_input_types())

  preds1 <- predict(result, new_data = test)
  tmp <- tempfile(fileext = "rds")
  saveRDS(result, tmp)
  rm(result); gc();
  result <- readRDS(tmp)
  preds2 <- predict(result, new_data = test)

  expect_equal(preds1, preds2)
})

test_that("can make rolling predictions", {

  init <- max(walmart_data()$Date) -lubridate::weeks(8)
  train <- walmart_data() %>%
    dplyr::filter(Date <= init)
  test <- walmart_data() %>%
    dplyr::filter(Date > init) %>%
    dplyr::filter(Store == 1, Dept == 1)

  result <- tft(walmart_recipe(), train, lookback = 120, horizon = 4,
                epochs = 1, input_types = walmart_input_types())

  predictions <- rolling_predict(result, past_data = train, new_data = test)
  expect_equal(nrow(predictions), 2)
  expect_equal(ncol(predictions), 3)
  for (p in predictions$.pred) {
    expect_equal(nrow(p), 4)
    expect_equal(ncol(p), 3)
  }

  test <- walmart_data() %>%
    dplyr::filter(Date > init) %>%
    dplyr::filter(Store %in% c(1,2), Dept == 1)

  predictions <- rolling_predict(result, past_data = train, new_data = test)
  expect_equal(nrow(predictions), 2)
  expect_equal(ncol(predictions), 3)
  for (p in predictions$.pred) {
    expect_equal(nrow(p), 8)
    expect_equal(ncol(p), 3)
  }
})



