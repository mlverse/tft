library(recipes)
library(luz)
library(zeallot)

data(walmart_sales, package = "walmartdata")
df <- walmart_sales %>%
  mutate(
    Store = as.character(Store),
    Dept = as.character(Dept)
  ) %>%
  dplyr::filter(Store %in% c(1, 2), Dept %in% c(1,2))%>%
  tsibble::tsibble(
    key = c(Store, Dept, Type, Size),
    index = Date
  ) %>%
  tsibble::group_by_key() %>%
  tsibble::fill_gaps(
    Weekly_Sales = 0,
    IsHoliday = FALSE
  ) %>%
  tidyr::fill(Size, Temperature, Fuel_Price, CPI, Unemployment, .direction = "down")

recipe <- recipe(Weekly_Sales ~ ., data = df) %>%
  update_role(!!!tsibble::key_vars(df), new_role = "key") %>%
  update_role(!!!tsibble::index_var(df), new_role = "index") %>%
  step_date(Date, role = "known", features = c("year", "month", "doy")) %>%
  update_role(IsHoliday, new_role = "unused") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_indicate_na(starts_with("MarkDown")) %>%
  step_impute_mean(starts_with("Markdown")) %>%
  step_include_roles()

result <- tft(recipe, df, lookback = 120, horizon = 4)
forecast(result)

rec <- prep(recipe)
dataset <- time_series_dataset(juice(rec), rec$term_info, lookback = 120, assess_stop = 4)
str(dataset[1])
n_features <- get_n_features(dataset[1][[1]])
hidden_state_size <- 16
feature_sizes <- dataset$feature_sizes

system.time({
  x <- coro::collect(torch::dataloader(dataset, batch_size = 256, shuffle = TRUE), 1)
})

m <- temporal_fusion_transformer(
  num_features = n_features,
  feature_sizes = feature_sizes,
  hidden_state_size = 16,
  dropout = 0.1,
  num_quantiles = 3,
  num_heads = 1
)

model <- temporal_fusion_transformer %>%
  luz::setup(
    loss = quantile_loss(quantiles = c(0.1, 0.5, 0.9)),
    optimizer = torch::optim_adam
  ) %>%
  luz::set_hparams(
    num_features = n_features,
    feature_sizes = feature_sizes,
    hidden_state_size = 16,
    dropout = 0.1,
    num_quantiles = 3,
    num_heads = 1
  ) %>%
  luz::set_opt_hparams(
    lr = 0.03
  ) %>%
  fit(dataset, epochs = 5, dataloader_options = list(
    batch_size = 256, num_workers = 0
  ))
