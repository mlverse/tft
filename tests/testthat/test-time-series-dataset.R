test_that("can create a spec", {

  init <- max(walmart_data()$Date) -lubridate::weeks(8)
  train <- walmart_data() %>%
    dplyr::filter(Date <= init)
  test <- walmart_data() %>%
    dplyr::filter(Date > init) %>%
    dplyr::filter(Store == 1, Dept == 1)

  spec <- tft_dataset_spec(walmart_recipe(), train)

  expect_snapshot(print(spec))

  spec <- tft_dataset_spec(walmart_recipe(), train) %>%
    spec_time_splits(lookback = 52, horizon = 4) %>%
    spec_covariate_index(Date) %>%
    spec_covariate_key(Store, Dept) %>%
    spec_covariate_static(Type, Size) %>%
    spec_covariate_known(starts_with("MarkDown"), starts_with("Date_"),
                         starts_with("na_ind"))

  expect_snapshot_output(print(spec))

  spec <- prep(spec)

  expect_snapshot_output(print(spec))
})

test_that("step  and subsample works", {

  init <- max(walmart_data()$Date) -lubridate::weeks(8)
  train <- walmart_data() %>%
    dplyr::filter(Date <= init)
  test <- walmart_data() %>%
    dplyr::filter(Date > init) %>%
    dplyr::filter(Store == 1, Dept == 1)

  spec <- tft_dataset_spec(walmart_recipe(), train) %>%
    spec_covariate_index(Date) %>%
    spec_covariate_key(Store, Dept) %>%
    spec_covariate_static(Type, Size) %>%
    spec_covariate_known(starts_with("MarkDown"), starts_with("Date_"),
                         starts_with("na_ind"))

  spec1 <- spec %>%
    spec_time_splits(lookback = 52, horizon = 4, step = 1) %>%
    prep()

  spec2 <- spec %>%
    spec_time_splits(lookback = 52, horizon = 4, step = 2) %>%
    prep()

  expect_equal(length(transform(spec1))/2, length(transform(spec2)))
  expect_equal(
    length(transform(spec1, subsample = 0.1))*10,
    length(transform(spec1))
  )
  expect_equal(
    length(transform(spec2, subsample = 0.1))*10,
    length(transform(spec2))
  )
})

