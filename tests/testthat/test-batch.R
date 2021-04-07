test_that("batch_data works with roles in vic_elec dataset", {
  library(recipes)
  skip_on_os("mac")

  data("vic_elec", package = "tsibbledata")
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

  expect_length(x, 11)
  expect_equal(x$known$numerics$size()[2:3], c( 20, 0))
  expect_equal(x$known$categorical$size()[2:3], c( 20, 1))
  expect_equal(x$observed$numerics$size()[2:3], c( 20, 1))
  expect_equal(x$observed$categorical$size()[2:3], c( 20, 0))
  expect_equal(x$static$numerics$size()[2:3], c( 20, 0))
  expect_equal(x$static$categorical$size()[2:3], c( 20, 1))
  expect_equal(x$target$numerics$size()[2:3], c( 20, 1))
  expect_equal(x$target$categorical$size()[2:3], c( 20, 0))
})
