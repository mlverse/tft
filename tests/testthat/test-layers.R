test_that("gated linear units works, w & wo dropout_rate", {

  x <- torch::torch_randn(100, 10)
  glu <- gated_linear_unit(10, 10)

  expect_equal( glu(x)[[1]]$shape, c(100, 10)  )

  glu <- gated_linear_unit(10, 10, dropout_rate=0.1)

  expect_equal( glu(x)[[1]]$shape, c(100, 10)  )

})

test_that("time distributed layer works", {

  x <- torch::torch_ones(5, 10, 15)
  linear <- torch::nn_linear(15, 1)
  td <- time_distributed(linear)

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

test_that("gated residual network, w and wo output_size, w or wo dropout_rate, w or wo return_gate", {

  x <- torch::torch_randn(32, 10)
  grn <- gated_residual_network(10, 5)

  expect_equal(grn(x)$shape, c(32, 5))

  grn <- gated_residual_network(10, 5, 3)

  expect_equal(grn(x)$shape, c(32, 3))

  grn <- gated_residual_network(10, 5, 3, dropout_rate=0.2)

  expect_equal(grn(x)$shape, c(32, 3))

  grn <- gated_residual_network(10, 5, 3, return_gate=TRUE)

  expect_equal(grn(x)[[1]]$shape, c(32, 3))
  expect_equal(grn(x)[[2]]$shape, c(32, 3))



})

test_that("time_distributed gated residual network & gated residual network wo time_distributed", {

  grn <- gated_residual_network(10, 5)
  x <- torch::torch_randn(32, 100, 10)
  td <- time_distributed(grn, 2)

  expect_equal(td(x)$shape, c(32, 100, 5))

  grn <- gated_residual_network(10, 5, use_time_distributed=FALSE)
  x <- torch::torch_randn(32, 100, 10)
  expect_equal(grn(x)$shape, c(32, 100, 5))


})
