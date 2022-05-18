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
    spec_covariate_keys(Store, Dept) %>%
    spec_covariate_static(Type, Size) %>%
    spec_covariate_known(starts_with("MarkDown"), starts_with("Date_"),
                         starts_with("na_ind"))

  expect_snapshot_output(print(spec))

  spec <- prep(spec)

  expect_snapshot_output(print(spec))
})

