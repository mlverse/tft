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
                epochs = 1)

  new_data <- future_data(result$past_data, 4)
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

  result <- tft(walmart_recipe(), walmart_data(), lookback = 120, horizon = 4,
                epochs = 1)

  preds <- forecast(result)
  expect_s3_class(preds, "tft_forecast")
  expect_equal(nrow(preds), 16)
  expect_equal(ncol(preds), 8)

})

test_that("can make full predictions", {

  result <- tft(walmart_recipe(), walmart_data(), lookback = 120, horizon = 4,
                epochs = 1)

  pred <- predict(
    result,
    mode = "full",
    new_data = walmart_data() %>% dplyr::filter(Store == 1)
  )

  expect_equal(nrow(pred), 200)
  expect_equal(ncol(pred), 28)
  expect_true(!is.null(pred$.pred_at))
})


