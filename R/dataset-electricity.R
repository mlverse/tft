electricity_dataset <- torch::dataset(
  "electricity_dataset",
  initialize = function(root) {
    self$root <- root
    self$download() # sets txt_path attribute
    self$preprocess() # sets the data attribute
  },
  download = function() {
    success("Download step:")
    self$txt_path <- electricity_download(self$root)
    success("Done: Download step!")
  },
  preprocess = function(txt_path) {
    success("Preprocessing step:")
    rds_path <- fs::path(fs::path_dir(txt_path), "preproc.rds")
    if (!fs::file_exists(rds_path)) {
      success("Preprocessing data...")
      data <- electricity_preprocess(self$txt_path)
      saveRDS(data, rds_path)
    } else {
      success("Skipping preprocessing. Reading data from disk...")
      data <- readRDS(rds_path)
    }
    self$data <- data
    sucess("Done: Preprocessing step!")
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
      cat_id   = id,
      cat_hour = hour,
      cat_day_of_week = day_of_week
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
