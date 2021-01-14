test_that("custom roles are working", {

  df <- tibble::tibble(
    id = runif(10),
    x = runif(10),
    z = runif(10)
  )

  rec <- recipes::recipe(x ~ ., df) %>%
    role_id(id) %>%
    role_static(id) %>%
    role_known(x) %>%
    role_observed(x) %>%
    role_time(z) %>%
    recipes::prep()

  expect_equal(
    names(recipes::bake(rec, df, all_id())),
    "id"
  )

  expect_equal(
    names(recipes::bake(rec, df, all_static())),
    "id"
  )

  expect_equal(
    names(recipes::bake(rec, df, all_known())),
    "x"
  )

  expect_equal(
    names(recipes::bake(rec, df, all_observed())),
    "x"
  )

  expect_equal(
    names(recipes::bake(rec, df, all_time())),
    "z"
  )



})
