test_that("batch", {

  skip_on_os("mac")

  elec <- electricity_dataset(root = "data-raw")
  x <- batch_data(elec$splits$train, elec$transform_inputs, 100)
  expect_length(x, 4)
})
