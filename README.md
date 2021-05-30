
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

Currently,`tft` relies on development versions of `mlverse/torch` and
`tidymodels/recipes`. You can install the development version
[GitHub](https://github.com/) with:

``` r
# install.packages("remotes")
remotes::install_github("mlverse/torch")
remotes::install_github("tidymodels/recipes")
remotes::install_github("mlverse/tft")
```

## Example

``` r
library(tft)
library(rsample)
library(recipes)
#> Loading required package: dplyr
#> 
#> Attaching package: 'dplyr'
#> The following objects are masked from 'package:stats':
#> 
#>     filter, lag
#> The following objects are masked from 'package:base':
#> 
#>     intersect, setdiff, setequal, union
#> 
#> Attaching package: 'recipes'
#> The following object is masked from 'package:stats':
#> 
#>     step
library(yardstick)
#> For binary classification, the first factor level is assumed to be the event.
#> Use the argument `event_level = "second"` to alter this as needed.
set.seed(1)

data("vic_elec", package = "tsibbledata")
vic_elec <- vic_elec[1:256,] %>% 
  mutate(Location = as.factor("Victoria")) 
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


fit <- tft_fit(rec, vic_elec_train, epochs = 15, batch_size=100, total_time_steps=12, num_encoder_steps=10, verbose=T )
#> [Epoch 001] Loss: 2560.704346
#> [Epoch 002] Loss: 2564.179688
#> [Epoch 003] Loss: 2552.926025
#> [Epoch 004] Loss: 2556.754639
#> [Epoch 005] Loss: 2547.451904
#> [Epoch 006] Loss: 2545.158325
#> [Epoch 007] Loss: 2538.207275
#> [Epoch 008] Loss: 2530.014282
#> [Epoch 009] Loss: 2523.856934
#> [Epoch 010] Loss: 2520.648315
#> [Epoch 011] Loss: 2514.970459
#> [Epoch 012] Loss: 2497.054443
#> [Epoch 013] Loss: 2492.264893
#> [Epoch 014] Loss: 2480.940186
#> [Epoch 015] Loss: 2468.159546

yhat <- predict(fit, rec, vic_elec_test)
```
