device <- torch::torch_device(if (torch::cuda_is_available()) "cuda" else "cpu")

test_that("gated linear units works, w & wo dropout_rate", {

  x <- torch::torch_randn(100, 10, device=device)
  glu <- gated_linear_unit(10, 10)$to(device=device)

  expect_equal( glu(x)[[1]]$shape, c(100, 10)  )

  glu <- gated_linear_unit(10, 10, dropout_rate=0.1)$to(device=device)

  expect_equal( glu(x)[[1]]$shape, c(100, 10)  )
})

test_that("time distributed layer works", {

  x <- torch::torch_ones(5, 10, 15, device=device)
  linear <- torch::nn_linear(15, 1)
  td <- time_distributed(linear)$to(device=device)

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

  x <- torch::torch_randn(32, 10, device=device)
  grn <- gated_residual_network(10, 5)$to(device=device)

  expect_equal(grn(x)$shape, c(32, 5))

  grn <- gated_residual_network(10, 5, 3)$to(device=device)

  expect_equal(grn(x)$shape, c(32, 3))

  grn <- gated_residual_network(10, 5, 3, dropout_rate=0.2)$to(device=device)

  expect_equal(grn(x)$shape, c(32, 3))

  grn <- gated_residual_network(10, 5, 3, return_gate=TRUE)$to(device=device)

  expect_equal(grn(x)[[1]]$shape, c(32, 3))
  expect_equal(grn(x)[[2]]$shape, c(32, 3))


})

test_that("gated residual network works in all initial conditions w context", {

  x <- torch::torch_randn(32, 10, device=device)
  context <- torch::torch_ones_like(x)$to(device=device)
  grn <- gated_residual_network(10, 5)$to(device=device)

  expect_equal(grn(x, context)$shape, c(32, 5))

  grn <- gated_residual_network(10, 5, 3)$to(device=device)

  expect_equal(grn(x, context)$shape, c(32, 3))

  grn <- gated_residual_network(10, 5, 3, dropout_rate=0.2)$to(device=device)

  expect_equal(grn(x, context)$shape, c(32, 3))

  grn <- gated_residual_network(10, 5, 3, return_gate=TRUE)$to(device=device)

  expect_equal(grn(x, context)[[1]]$shape, c(32, 3))
  expect_equal(grn(x, context)[[2]]$shape, c(32, 3))


})

test_that("time_distributed gated residual network & gated residual network wo time_distributed", {

  grn <- gated_residual_network(10, 5)
  x <- torch::torch_randn(32, 100, 10, device=device)
  td <- time_distributed(grn, 2)$to(device=device)

  expect_equal(td(x)$shape, c(32, 100, 5))

  grn <- gated_residual_network(10, 5, use_time_distributed=FALSE)$to(device=device)
  x <- torch::torch_randn(32, 100, 10, device=device)
  expect_equal(grn(x)$shape, c(32, 100, 5))


})

test_that("scaled_dot_product_attention works, w or wo mask", {

  sdp_attention <- scaled_dot_product_attention()$to(device=device)
  query <- torch::nn_linear(10, 20, bias=FALSE)$to(device=device)
  key <- torch::nn_linear(10, 20, bias=FALSE)$to(device=device)
  value <- torch::nn_linear(10, 20, bias=FALSE)$to(device=device)

  #without mask
  mask <- NULL
  x <- torch::torch_randn(2, 4, 10, device=device)

  output_attn_lst <- sdp_attention(query(x), key(x), value(x), mask)
  output <- output_attn_lst[[1]]
  attn <- output_attn_lst[[2]]

  expect_equal(output$shape, c(2, 4, 20))
  expect_equal(attn$shape, c(2, 4, 4))

  # with mask
  mask <- array(as.numeric(rnorm(2*4*4)< 1), dim=c(2,4,4))

  output_attn_lst <- sdp_attention(query(x), key(x), value(x), mask)
  output <- output_attn_lst[[1]]
  attn <- output_attn_lst[[2]]

  expect_equal(output$shape, c(2, 4, 20))
  expect_equal(attn$shape, c(2, 4, 4))

})

test_that("interpretable_multihead_attention works", {

  multihead_attn <- interpretable_multihead_attention(n_head=2, d_model=12, dropout_rate=0)$to(device=device)
  # nn shape test
  expect_length(multihead_attn$modules, 12)
  expect_length(multihead_attn$attention$modules, 3)
  expect_length(multihead_attn$attention$modules[[1]], 1)
  expect_length(multihead_attn$attention$modules[[2]], 1)

  #without mask
  mask <- NULL
  query <- key <- value <- torch::torch_randn(5, 4, 12, device=device)
  x <- torch::torch_randn(5, 4, 10, device=device)

  outputs_attn_lst <- multihead_attn(query, key, value, mask)
  outputs <- outputs_attn_lst[[1]]
  attn <- outputs_attn_lst[[2]]

  expect_equal(outputs$shape, c(5, 4, 12))
  expect_equal(attn$shape, c(2, 5, 4, 4))

  # with mask
  mask <- array(as.numeric(rnorm(5*4*4)< 1), dim=c(5,4,4))

  outputs_attn_lst <- multihead_attn(query, key, value, mask)
  outputs <- outputs_attn_lst[[1]]
  attn <- outputs_attn_lst[[2]]

  expect_equal(outputs$shape, c(5, 4, 12))
  expect_equal(attn$shape, c(2, 5, 4, 4))

})

test_that("static_combine_and_mask works", {

  static_cbn_n_mask <- static_combine_and_mask(10, num_static=5, hidden_layer_size=7, dropout_rate=0)$to(device=device)
  embedding <- torch::torch_ones(c(4, 5, 7), device=device)
  #without additional_context
  additional_context <- NULL


  static_vec_sparse_weights <- static_cbn_n_mask(embedding, additional_context)
  static_vec <- static_vec_sparse_weights[[1]]
  sparse_weights <- static_vec_sparse_weights[[2]]

  expect_equal(static_vec$shape, c(4,7))
  expect_equal(sparse_weights$shape, c(4,5,1))

  # with additional_context (like flatten_embeddings = [?, num_static*hidden_layer])
  additional_context <- array(as.numeric(rnorm(4*5*7)< 1), dim=c(4,35)) %>% torch::torch_tensor(device = device)

  static_vec_sparse_weights <- static_cbn_n_mask(embedding, additional_context)
  static_vec <- static_vec_sparse_weights[[1]]
  sparse_weights <- static_vec_sparse_weights[[2]]

  expect_equal(static_vec$shape, c(4,7))
  expect_equal(sparse_weights$shape, c(4,5,1))

})

test_that("lstm_combine_and_mask works", {

  lstm_cbn_n_mask <- lstm_combine_and_mask(10, num_inputs=5, hidden_layer_size=7, dropout_rate=0)$to(device=device)
  embedding <- torch::torch_ones(c(4, 24, 7, 5), device=device)
  #without additional_context
  additional_context <- NULL


  lstm_output_lst <- lstm_cbn_n_mask(embedding, additional_context)
  temporal_ctx <- lstm_output_lst[[1]]
  sparse_weights <- lstm_output_lst[[2]]
  static_gate <- lstm_output_lst[[3]]

  expect_equal(temporal_ctx$shape, c(4,24,7))
  expect_equal(sparse_weights$shape, c(4,24,1, 5))
  expect_equal(static_gate, NULL)

  # with additional_context (like flatten_embeddings = [?, num_static*hidden_layer])
  additional_context <- array(as.numeric(rnorm(4*7*5)< 1), dim=c(4,35)) %>% torch::torch_tensor(device = device)

  lstm_output_lst <- lstm_cbn_n_mask(embedding, additional_context)
  temporal_ctx <- lstm_output_lst[[1]]
  sparse_weights <- lstm_output_lst[[2]]
  static_gate <- lstm_output_lst[[3]]

  expect_equal(temporal_ctx$shape, c(4,24,7))
  expect_equal(sparse_weights$shape, c(4,24,1, 5))
  expect_equal(static_gate$shape, c(4,24, 5))

})



