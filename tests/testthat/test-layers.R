test_that("gated linear units works", {

  x <- torch::torch_randn(100, 10)
  glu <- gated_linear_unit(10)

  expect_equal(
    glu(x)$shape,
    c(100, 10)
  )

})

test_that("time distributed layer works", {

  x <- torch::torch_ones(5, 10, 15)
  linear <- torch::nn_linear(15, 1)
  td <- time_distributed(linear, dim = 1)

  expect_equal(
    td(x)$shape,
    c(5, 10, 1)
  )

  o <- td(x)
  expect_true(
    torch::torch_equal(o[1,..], o[2,..])
  )
  expect_true(
    torch::torch_equal(o[1,..], o[5,..])
  )

})
