test_that("tft_fit works only for recipe", {
  expect_error(tft_fit(iris[,-5], iris$Species),
               regexp = "is not defined for")
})
