test_that("first fit", {
  expect_error(
    result <- tft(walmart_recipe(), walmart_data(), lookback = 120, horizon = 4,
                  epochs = 1, input_types = walmart_input_types()),
    regexp = NA
  )
})
