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




