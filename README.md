
<!-- README.md is generated from README.Rmd. Please edit that file -->

# tft

<!-- badges: start -->

[![R build
status](https://github.com/mlverse/tft/workflows/R-CMD-check/badge.svg)](https://github.com/mlverse/tft/actions)
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
[![CRAN
status](https://www.r-pkg.org/badges/version/tft)](https://CRAN.R-project.org/package=tft)
[![](https://cranlogs.r-pkg.org/badges/tft)](https://cran.r-project.org/package=tft)
[![Codecov test
coverage](https://codecov.io/gh/mlverse/tft/branch/master/graph/badge.svg)](https://codecov.io/gh/mlverse/tft?branch=master)

<!-- badges: end -->

An R implementation of: [tft: Temporal Fusion
Transformer](https://arxiv.org/pdf/1912.09363.pdf). The code in this
repository is an R port of
[akeskiner/Temporal\_Fusion\_Transform](https://github.com/akeskiner/Temporal_Fusion_Transform)
PyTorch’s implementation using the
[torch](https://github.com/mlverse/torch) package.

## Installation

You can install the development version [GitHub](https://github.com/)
with:

``` r
# install.packages("remotes")
remotes::install_github("mlverse/tft")
```

## Example

``` r
library(tft)
library(rsample)
suppressMessages(library(recipes))
suppressMessages(library(yardstick))
suppressMessages(library(tsibble))
set.seed(1)

data("vic_elec", package = "tsibbledata")
vic_elec <- vic_elec %>% 
  mutate(Location = as.factor("Victoria"))

str(vic_elec)
#> tbl_ts [52,608 × 6] (S3: tbl_ts/tbl_df/tbl/data.frame)
#>  $ Time       : POSIXct[1:52608], format: "2012-01-01 00:00:00" "2012-01-01 00:30:00" ...
#>  $ Demand     : num [1:52608] 4383 4263 4049 3878 4036 ...
#>  $ Temperature: num [1:52608] 21.4 21.1 20.7 20.6 20.4 ...
#>  $ Date       : Date[1:52608], format: "2012-01-01" "2012-01-01" ...
#>  $ Holiday    : logi [1:52608] TRUE TRUE TRUE TRUE TRUE TRUE ...
#>  $ Location   : Factor w/ 1 level "Victoria": 1 1 1 1 1 1 1 1 1 1 ...
#>  - attr(*, "key")= tibble [1 × 1] (S3: tbl_df/tbl/data.frame)
#>   ..$ .rows: list<int> [1:1] 
#>   .. ..$ : int [1:52608] 1 2 3 4 5 6 7 8 9 10 ...
#>   .. ..@ ptype: int(0) 
#>  - attr(*, "index")= chr "Time"
#>   ..- attr(*, "ordered")= logi TRUE
#>  - attr(*, "index2")= chr "Time"
#>  - attr(*, "interval")= interval [1:1] 30m
#>   ..@ .regular: logi TRUE
```

``` r
vic_elec_split <- initial_time_split(vic_elec, prop=3/4, lag=96)
  
vic_elec_train <- training(vic_elec_split)
vic_elec_test <- testing(vic_elec_split)

rec <- recipe(Demand ~ ., data = vic_elec_train) %>%
  update_role(Date, new_role="id") %>%
  update_role(Time, new_role="time") %>%
  update_role(Temperature, new_role="observed_input") %>%
  update_role(Holiday, new_role="known_input") %>%
  update_role(Location, new_role="static_input") %>%
  step_normalize(all_numeric(), -all_outcomes())


fit <- tft_fit(rec, vic_elec_train, epochs = 100, batch_size=100, total_time_steps=12, num_encoder_steps=10, verbose=TRUE)

yhat <- predict(fit, rec, vic_elec_test)
```
