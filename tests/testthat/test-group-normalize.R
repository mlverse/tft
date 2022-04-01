test_that("group normalize", {
  x <- tibble(
    g = c(rep("a", 5), rep("b", 5)),
    h = rep(c("a", "b"), 5),
    v1 = runif(10),
    v2 = runif(10)
  )

  rec <- recipes::recipe(~ ., x) %>%
    step_group_normalize(
      v1, v2,
      groups = c(g)
    ) %>%
    recipes::prep()

  out <- recipes::juice(rec)

  expect_equal(out$v1[1:5], as.numeric(scale(x$v1[1:5])))
  expect_equal(out$v2[1:5], as.numeric(scale(x$v2[1:5])))
  expect_equal(out$v1[6:10], as.numeric(scale(x$v1[6:10])))
  expect_equal(out$v2[6:10], as.numeric(scale(x$v2[6:10])))

  out <- bake(rec, x)

  expect_equal(out$v1[1:5], as.numeric(scale(x$v1[1:5])))
  expect_equal(out$v2[1:5], as.numeric(scale(x$v2[1:5])))
  expect_equal(out$v1[6:10], as.numeric(scale(x$v1[6:10])))
  expect_equal(out$v2[6:10], as.numeric(scale(x$v2[6:10])))
})
