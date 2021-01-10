#' @include utils.R

electricity_dataset <- torch::dataset(
  "electricity_dataset",
  initialize = function(root, valid_boundary = 1315, test_boundary = 1339) {
    self$root <- root
    self$download() # sets txt_path attribute
    self$preprocess() # sets data attribute
    self$split(valid_boundary, test_boundary) # sets the splits attribute
    self$set_recipes() # sets the recipes attribute
    self$transform_splits()
  },
  download = function() {
    success("Download step:")
    self$txt_path <- electricity_download(self$root)
    success("Done: Download step!")
  },
  preprocess = function() {
    success("Preprocessing step:")
    rds_path <- fs::path(fs::path_dir(self$txt_path), "preproc.rds")
    if (!fs::file_exists(rds_path)) {
      success("Preprocessing data...")
      data <- electricity_preprocess(self$txt_path)
      saveRDS(data, rds_path)
    } else {
      success("Skipping preprocessing. Reading data from disk...")
      data <- readRDS(rds_path)
    }
    self$data <- data
    success("Done: Preprocessing step!")
  },
  split = function(valid_boundary, test_boundary) {
    success("Spliting step:")
    self$splits <- electricity_split(self$data, valid_boundary, test_boundary)
    success("Done: Spliting step!")
  },
  set_recipes = function() {
    success("Creating recipes step:")
    self$recipes <- electricity_recipe(self$splits$train)
    success("Done: Creating recipes!")
  },
  transform_inputs = function(data) {
    electricity_transform_input(self$recipes, data)
  },
  transform_splits = function() {
    success("Transforming the splits!")
    self$splits <- lapply(self$splits, self$transform_inputs)
    success("Done: Transforming the splits!")
  },
  .length = function() {
    NA
  }
)

electricity_preprocess <- function(txt_path) {

  # warns because the csv doesn't have colnames for the first column.
  suppressWarnings({
    el <- readr::read_csv2(
      file = txt_path,
      col_types = readr::cols(
        .default = readr::col_double(),
        X1 = readr::col_datetime(format = "")
      ),
      locale = readr::locale(decimal_mark = ",", grouping_mark = ".")
    )
  })

  success("Aggregating into hourly data...")
  el_lng <- el %>%
    dplyr::rename(date = X1) %>%
    tidyr::pivot_longer(
      cols = tidyr::starts_with("MT_"),
      names_to = "serie",
      values_to = "pwr_usage"
    )

  el_hour <- el_lng %>%
    dplyr::filter(pwr_usage > 0) %>%
    padr::thicken("hour") %>%
    dplyr::group_by(serie, date = date_hour) %>%
    dplyr::summarise(
      pwr_usage = sum(pwr_usage),
      .groups = "drop"
    ) %>%
    padr::pad(
      interval = "hour",
      group = "serie",
      break_above = 100
    ) %>%
    dplyr::mutate(pwr_usage = dplyr::coalesce(pwr_usage, 0))

  success("Computing features...")
  earliest_time <- min(el_hour$date)
  el_hour <- el_hour %>%
    dplyr::mutate(
      t                = as.integer(difftime(date, earliest_time, units = "hours")),
      days_from_start  = as.integer(difftime(date, earliest_time, units = "days")),
      id               = as.integer(as.factor(serie)),
      hour             = lubridate::hour(date),
      day              = lubridate::day(date),
      day_of_week      = lubridate::wday(date),
      month            = lubridate::month(date),
      hours_from_start = t,
      cat_id   = as.factor(id),
      cat_hour = as.factor(hour),
      cat_day_of_week = as.factor(day_of_week)
    )

  success("Filtering to match other research papers...")
  el_filt <- el_hour %>%
    dplyr::filter(
      days_from_start >= 1096,
      days_from_start < 1346
    )

  el_hour
}

electricity_download <- function(root) {
  url <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

  zip_path <- fs::path(root, "LD2011_2014.txt.zip")
  if (!fs::file_exists(zip_path)) {
    success("Downloading data... \n")

    if (!fs::dir_exists(root))
      fs::dir_create(root)

    download.file(url, destfile = zip_path)

    if (tools::md5sum(zip_path) != "0f0fd085f742acd4f36687e7c071e521") {
      rlang::abort(paste(
        "Downloaded file doesn't have the reported md5sum.",
        "Expected: '0f0fd085f742acd4f36687e7c071e521' and ",
        "got: '", tools::md5sum(zip_path), "'."))
    }

  } else {
    success("Skipping download. Zip file already exists.")
  }

  txt_path <- fs::path(root, "LD2011_2014.txt")
  if (!fs::file_exists(txt_path)) {
    success("Unzipping...\n")
    unzip(zip_path, exdir = root)

    if (tools::md5sum(txt_path) != "e317add771cebd2df4121e50570eaa25") {
      rlang::abort(paste(
        "Something wrong happened while unzipping. Unexpected md5 hash.",
        "Expected: 'e317add771cebd2df4121e50570eaa25' and ",
        "got: '", tools::md5sum(txt_path), "'."))
    }
  } else {
    success("Skipping unziping. Text file already exists.")
  }

  txt_path
}

electricity_split <- function(data, valid_boundary = 1315, test_boundary = 1339) {
  # TODO: understand and explain the -7 stuff.
  list(
    train = dplyr::filter(data, days_from_start < valid_boundary),
    valid = dplyr::filter(
      data,
      days_from_start >= (valid_boundary - 7),
      days_from_start < test_boundary
    ),
    test  = dplyr::filter(data, days_from_start >= (test_boundary - 7))
  )
}

electricity_recipe <- function(data) {

  base_recipe <- recipes::recipe(
    pwr_usage ~ .,
    data = data %>% dplyr::select(-serie)
  ) %>%
    recipes::update_role(
      id,
      new_role = input_types$id
    ) %>%
    recipes::update_role(
      hour, day_of_week, hours_from_start,
      new_role = input_types$known_input
    ) %>%
    recipes::update_role(
      t,
      new_role = input_types$time
    ) %>%
    recipes::update_role(
      cat_id,
      new_role = input_types$static_input
    )

  num_recipe <- base_recipe %>%
    recipes::step_normalize(
      recipes::all_outcomes()
    ) %>%
    recipes::step_normalize(
      recipes::has_role("known_input"),
      -recipes::all_nominal()
    )

  success("Creating numeric recipes")
  num_recipes <- data %>%
    dplyr::group_nest(serie) %>%
    dplyr::mutate(recipe = purrr::map(data, ~recipes::prep(num_recipe, .x))) %>%
    dplyr::select(-data)

  success("Creating categorical recipe")
  cat_recipe <- base_recipe %>%
    recipes::step_unknown(
      recipes::has_role("static_input"),
      recipes::has_role("known_input"),
      -recipes::all_numeric()
    ) %>%
    recipes::prep()

  list(
    num_recipes = num_recipes,
    cat_recipe = cat_recipe
  )
}

electricity_transform_input <- function(recipes, data) {

  nested_data <- data %>%
    dplyr::group_nest(serie) %>%
    dplyr::left_join(recipes$num_recipes, by = "serie")

  if (any(sapply(nested_data$recipe, is.na)))
    rlang::abort(paste0("No numeric scaler found for at least one serie"))

  data <- purrr::map2_dfr(
    nested_data$data,
    nested_data$recipe,
    ~recipes::bake(.y, .x)
  )

  data <- recipes::bake(recipes$cat_recipe, data)
  data
}
