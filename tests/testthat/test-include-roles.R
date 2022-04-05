test_that("include role works", {
  rec <- recipes::recipe(
    mpg ~.,
    data = mtcars %>% dplyr::mutate(hp = as.character(hp))
  ) %>%
    recipes::update_role(am, vs, new_role = "index") %>%
    recipes::update_role(disp, hp, new_role = "hello") %>%
    step_include_roles() %>%
    recipes::prep()

  x <- bake(rec, mtcars)
  roles <- attr(x, "roles")

  expect_true(tibble::is_tibble(roles))
  expect_equal(roles$tft_role[roles$variable %in% c("am", "vs")], rep("index", 2))
  expect_equal(roles$role[roles$variable %in% c("am", "vs")], rep("predictor", 2))
  expect_equal(roles$tft_role[roles$variable %in% c("disp", "hp")], rep("hello", 2))

  x <- recipes::juice(rec)
  roles <- attr(x, "roles")

  expect_true(tibble::is_tibble(roles))
  expect_equal(roles$tft_role[roles$variable %in% c("am", "vs")], rep("index", 2))
  expect_equal(roles$role[roles$variable %in% c("am", "vs")], rep("predictor", 2))
  expect_equal(roles$tft_role[roles$variable %in% c("disp", "hp")], rep("hello", 2))
})

