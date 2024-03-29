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
