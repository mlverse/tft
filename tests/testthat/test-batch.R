test_that("batch_data works with roles in vic_elec dataset", {
  library(recipes)
  library(tsibbledata)
  skip_on_os("mac")

  data("vic_elec")
  vic_elec <- vic_elec[1:151,] %>%
    dplyr::mutate(Location = as.factor("Victoria"))
  rec <- recipe(Demand ~ ., data = vic_elec) %>%
    update_role(Date, new_role="id") %>%
    update_role(Time, new_role="time") %>%
    update_role(Temperature, new_role="observed_input") %>%
    update_role(Holiday, new_role="known_input") %>%
    update_role(Location, new_role="static_input") %>%
    step_normalize(all_numeric(), -all_outcomes())

  x <- batch_data(recipe=rec, df=vic_elec, total_time_steps=10, device="cpu")

  expect_length(x, 13)
  # test tensor shape is1 0 time_steps
  expect_equal(x$known$numerics$size()[2:3], c( 10, 0))
  expect_equal(x$known$categorical$size()[2:3], c( 10, 1))
  expect_equal(x$observed$numerics$size()[2:3], c( 10, 1))
  expect_equal(x$observed$categorical$size()[2:3], c( 10, 0))
  expect_equal(x$static$numerics$size()[2:3], c( 10, 0))
  expect_equal(x$static$categorical$size()[2:3], c( 10, 1))
  expect_equal(x$target$numerics$size()[2:3], c( 10, 1))
  expect_equal(x$target$categorical$size()[2:3], c( 10, 0))
  expect_s3_class(x$blueprint, c("default_recipe_blueprint","recipe_blueprint","hardhat_blueprint" ))
})


test_that("tft_initialize works with roles in vic_elec dataset", {
  library(recipes)
  library(tsibbledata)
  skip_on_os("mac")

  data("vic_elec")
  vic_elec <- vic_elec[1:151,] %>%
    dplyr::mutate(Location = as.factor("Victoria"))
  rec <- recipe(Demand ~ ., data = vic_elec) %>%
    update_role(Date, new_role="id") %>%
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
  expect_length(tft_model_lst$config, 24)
  expect_match(tft_model_lst$config$loss, "quantile_loss")
  expect_s3_class(tft_model_lst$network, c("tft","nn_module" ))
  expect_s3_class(tft_model_lst$network$output_layer, c("linear_layer","nn_module" ))
  expect_s3_class(tft_model_lst$network$static_context_variable_selection_grn, c("gated_residual_network","nn_module" ))
  expect_s3_class(tft_model_lst$network$static_context_enrichment_grn, c("gated_residual_network","nn_module" ))
  expect_s3_class(tft_model_lst$network$static_context_state_h_grn, c("gated_residual_network","nn_module" ))
  expect_s3_class(tft_model_lst$network$static_context_state_c_grn, c("gated_residual_network","nn_module" ))
  expect_s3_class(tft_model_lst$network$static_enrichment_grn, c("gated_residual_network","nn_module" ))
  expect_s3_class(tft_model_lst$network$decoder_grn, c("gated_residual_network","nn_module" ))
})


test_that("tft_initialize works with pinball_loss", {
  library(recipes)
  library(tsibbledata)
  skip_on_os("mac")

  data("vic_elec")
  vic_elec <- vic_elec[1:151,] %>%
    dplyr::mutate(Location = as.factor("Victoria"))
  rec <- recipe(Demand ~ ., data = vic_elec) %>%
    update_role(Date, new_role="id") %>%
    update_role(Time, new_role="time") %>%
    update_role(Temperature, new_role="observed_input") %>%
    update_role(Holiday, new_role="known_input") %>%
    update_role(Location, new_role="static_input") %>%
    step_normalize(all_numeric(), -all_outcomes())

  processed <- tft:::batch_data(recipe=rec, df=vic_elec, total_time_steps=10, device="cpu")
  config <- tft_config( epochs = 30, total_time_steps=12, num_encoder_steps=10, loss="pinball_loss")

  tft_model_lst <- tft:::tft_initialize(processed, config)
  expect_length(tft_model_lst, 5)
  # test tensor shape (20 time_steps while total_time_steps is hardcoded in hours)
  expect_match(tft_model_lst$config$loss, "pinball_loss")
})


test_that("tft_nn works with a small example inspired from README with tsibbledata::vic_elec", {
  nn <- tft:::tft_nn(input_dim = 5, output_dim = 1, cat_idx = c(5,6), cat_dims = list(2,1),
                     observed_idx = 3, static_idx = 6, target_idx = 2,
                     known_idx = 5, dropout_rate = 0, num_heads = 3,
                     total_time_steps = 10, num_encoder_steps = 8)

  expect_error(nn(
                 known_numerics = torch::torch_randn(100, 10, 0),
                 known_categorical = torch::torch_randint(1,3,size = c(100, 10, 1)),
                 observed_numerics = torch::torch_randn(100, 10, 1),
                 observed_categorical = torch::torch_randint(0,1,size = c(100, 10, 0)),
                 static_numerics = torch::torch_randn(100, 10, 0),
                 static_categorical = torch::torch_randint(1,2,size = c(100, 10, 1)),
                 target_numerics = torch::torch_randn(100, 10, 1),
                 target_categorical = torch::torch_randint(0,1,size = c(100, 10, 0))
               ),
               regexp=NA)

})


test_that("tft_train works with pure nominal inputs", {
  library(recipes)
  library(tsibbledata)
  skip_on_os("mac")

  data("vic_elec")
  vic_elec <- vic_elec[1:151,] %>%
    dplyr::mutate(Location = as.factor("Victoria"),
                  Temperature = factor(ceiling(Temperature), ordered = T))
  rec <- recipe(Demand ~ ., data = vic_elec) %>%
    update_role(Date, new_role="id") %>%
    update_role(Time, new_role="time") %>%
    update_role(Temperature, new_role="observed_input") %>%
    update_role(Holiday, new_role="known_input") %>%
    update_role(Location, new_role="static_input") %>%
    step_normalize(all_numeric(), -all_outcomes())

  processed <- tft:::batch_data(recipe=rec, df=vic_elec, total_time_steps=10, device="auto")
  config <- tft:::tft_config(batch_size=50, epochs = 3, total_time_steps=12, num_encoder_steps=10)

  tft_model_lst <- tft:::tft_initialize(processed, config)
  tft_model <-  tft:::new_tft_fit(tft_model_lst, blueprint = processed$blueprint)
  epoch_shift <- 0L
  expect_error(fit_lst <- tft:::tft_train(obj=tft_model, data=processed, config = config, epoch_shift),
               regexp=NA)

})


test_that("tft_train works with pure numerical inputs", {
  library(recipes)
  library(tsibbledata)
  skip_on_os("mac")

  data("vic_elec")
  vic_elec <- vic_elec[1:151,] %>%
    dplyr::mutate(Location = 1.5,
                  Holiday = lubridate::wday(Date))
  rec <- recipe(Demand ~ ., data = vic_elec) %>%
    update_role(Date, new_role="id") %>%
    update_role(Time, new_role="time") %>%
    update_role(Temperature, new_role="observed_input") %>%
    update_role(Holiday, new_role="known_input") %>%
    update_role(Location, new_role="static_input") %>%
    step_normalize(all_numeric(), -all_outcomes())

  processed <- tft:::batch_data(recipe=rec, df=vic_elec, total_time_steps=10, device="auto")
  config <- tft:::tft_config(batch_size=50, epochs = 3, total_time_steps=12, num_encoder_steps=10)

  tft_model_lst <- tft:::tft_initialize(processed, config)
  tft_model <-  tft:::new_tft_fit(tft_model_lst, blueprint = processed$blueprint)
  epoch_shift <- 0L
  expect_error(fit_lst <- tft:::tft_train(obj=tft_model, data=processed, config = config, epoch_shift),
               regexp=NA)

})

