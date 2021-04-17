test_that("batch_data works with roles in vic_elec dataset", {
  library(recipes)
  library(tsibbledata)
  skip_on_os("mac")

  data("vic_elec")
  vic_elec <- vic_elec %>%
    dplyr::mutate(Location = as.factor("Victoria")) %>%
    dplyr::rename(id = Date)
  rec <- recipe(Demand ~ ., data = vic_elec) %>%
    update_role(id, new_role="id") %>%
    update_role(Time, new_role="time") %>%
    update_role(Temperature, new_role="observed_input") %>%
    update_role(Holiday, new_role="known_input") %>%
    update_role(Location, new_role="static_input") %>%
    step_normalize(all_numeric(), -all_outcomes())

  x <- batch_data(recipe=rec, df=vic_elec, total_time_steps=10, device="cpu")

  expect_length(x, 12)
  # test tensor shape (20 time_steps while total_time_steps is hardcoded in hours)
  expect_equal(x$known$numerics$size()[2:3], c( 20, 0))
  expect_equal(x$known$categorical$size()[2:3], c( 20, 1))
  expect_equal(x$observed$numerics$size()[2:3], c( 20, 1))
  expect_equal(x$observed$categorical$size()[2:3], c( 20, 0))
  expect_equal(x$static$numerics$size()[2:3], c( 20, 0))
  expect_equal(x$static$categorical$size()[2:3], c( 20, 1))
  expect_equal(x$target$numerics$size()[2:3], c( 20, 1))
  expect_equal(x$target$categorical$size()[2:3], c( 20, 0))
  expect_s3_class(x$blueprint, c("default_recipe_blueprint","recipe_blueprint","hardhat_blueprint" ))
})


test_that("tft_initialize works with roles in vic_elec dataset", {
  library(recipes)
  library(tsibbledata)
  skip_on_os("mac")

  data("vic_elec")
  vic_elec <- vic_elec %>%
    dplyr::mutate(Location = as.factor("Victoria")) %>%
    dplyr::rename(id = Date)
  rec <- recipe(Demand ~ ., data = vic_elec) %>%
    update_role(id, new_role="id") %>%
    update_role(Time, new_role="time") %>%
    update_role(Temperature, new_role="observed_input") %>%
    update_role(Holiday, new_role="known_input") %>%
    update_role(Location, new_role="static_input") %>%
    step_normalize(all_numeric(), -all_outcomes())

  processed <- batch_data(recipe=rec, df=vic_elec, total_time_steps=10, device="cpu")
  config <- tft_config( epochs = 30, total_time_steps=12, num_encoder_steps=10)

  tft_model_lst <- tft_initialize(processed, config)
  expect_length(tft_model_lst, 5)
  # test tensor shape (20 time_steps while total_time_steps is hardcoded in hours)
  expect_length(tft_model_lst$metrics, 0)
  expect_length(tft_model_lst$checkpoints, 0)
  expect_length(tft_model_lst$config, 23)
  expect_s3_class(tft_model_lst$network, c("tft","nn_module" ))
  expect_s3_class(tft_model_lst$network$output_layer, c("linear_layer","nn_module" ))
  expect_s3_class(tft_model_lst$network$static_context_variable_selection_grn, c("gated_residual_network","nn_module" ))
  expect_s3_class(tft_model_lst$network$static_context_enrichment_grn, c("gated_residual_network","nn_module" ))
  expect_s3_class(tft_model_lst$network$static_context_state_h_grn, c("gated_residual_network","nn_module" ))
  expect_s3_class(tft_model_lst$network$static_context_state_c_grn, c("gated_residual_network","nn_module" ))
  expect_s3_class(tft_model_lst$network$static_enrichment_grn, c("gated_residual_network","nn_module" ))
  expect_s3_class(tft_model_lst$network$decoder_grn, c("gated_residual_network","nn_module" ))
})


