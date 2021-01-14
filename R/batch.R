batch_data <- function(df, time_steps) {

  time_steps <- 100

  positions <- df %>%
    dplyr::group_by(id) %>%
    dplyr::filter(dplyr::n() >= time_steps) %>%
    dplyr::group_split(.keep = TRUE) %>%
    purrr::map_dfr(
      ~tibble::tibble(
        id = dplyr::first(.x$id),
        start_time = seq(
          from = min(.x$date),
          # TODO dont hardcode hours here
          to   = max(.x$date) - lubridate::hours(time_steps),
          by   = "hours"
        ),
        end_time = start_time + lubridate::hours(time_steps)
      )
    )

  positions <- dplyr::sample_n(positions, 500)

  output <- positions %>%
    dplyr::group_split(id, start_time) %>%
    purrr::map(
      ~df %>%
        dplyr::filter(
          id == .x$id,
          date >= .x$start_time,
          date < .x$end_time
        )
    )

  tensors <- output %>%
    purrr::map(
      ~.x %>%
        dplyr::select(
          hours_from_start,
          days_from_start,
          hour,
          day_of_week,
          month
        ) %>%
        df_to_tensor()
    ) %>%
    torch::torch_stack()

}

df_to_tensor <- function(df) {
  df %>%
    dplyr::mutate(dplyr::across(where(is.factor), as.integer)) %>%
    as.matrix() %>%
    torch::torch_tensor()
}




