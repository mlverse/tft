test_that("include role works", {
  rec <- recipes::recipe(
    mpg ~.,
    data = mtcars %>% dplyr::mutate(hp = as.character(hp))
  ) %>%
    recipes::update_role(am, vs, new_role = "hello") %>%
    recipes::update_role(disp, hp, new_role = "world") %>%
    step_include_roles() %>%
    recipes::prep()
  x <- bake(rec, mtcars)
  roles <- attr(x, "roles")
  expect_true(tibble::is_tibble(roles))
  expect_equal(roles$role[roles$variable %in% c("am", "vs")], rep("hello", 2))
  expect_equal(roles$role[roles$variable %in% c("disp", "hp")], rep("world", 2))
})

test_that("works for modling", {

  rec <- recipes::recipe(mpg ~., data = mtcars) %>%
    recipes::update_role(am, vs, new_role = "hello") %>%
    recipes::update_role(disp, hp, new_role = "world") %>%
    step_include_roles()

  tft(rec, mtcars)

})
