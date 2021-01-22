batch_data <- function(df, transform, time_steps = 100) {

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

  known <- list(
    numerics = output %>%
      purrr::map(
        ~.x %>%
          transform(all_known() & recipes::all_numeric()) %>%
          df_to_tensor()
      ) %>%
      torch::torch_stack(),
    categorical = output %>%
      purrr::map(
        ~.x %>%
          transform(all_known() & recipes::all_nominal()) %>%
          df_to_tensor()
      ) %>%
      torch::torch_stack()
  )

  observed <- list(
    numerics = output %>%
      purrr::map(
        ~.x %>%
          transform(all_observed() & recipes::all_numeric()) %>%
          df_to_tensor()
      ) %>%
      torch::torch_stack(),
    categorical = output %>%
      purrr::map(
        ~.x %>%
          transform(all_observed() & recipes::all_nominal()) %>%
          df_to_tensor()
      ) %>%
      torch::torch_stack()
  )

  target <- list(
    numerics = output %>%
      purrr::map(
        ~.x %>%
          transform(recipes::all_outcomes() & recipes::all_numeric()) %>%
          df_to_tensor()
      ) %>%
      torch::torch_stack(),
    categorical = output %>%
      purrr::map(
        ~.x %>%
          transform(recipes::all_outcomes() & recipes::all_nominal()) %>%
          df_to_tensor()
      ) %>%
      torch::torch_stack()
  )

  static <- list(
    numerics = output %>% purrr::map(
      ~.x %>%
        transform(all_static() & recipes::all_nominal()) %>%
        df_to_tensor()
    ) %>%
      torch::torch_stack(),
    categorical = output %>% purrr::map(
      ~.x %>%
        transform(all_static() & recipes::all_nominal()) %>%
        df_to_tensor()
    ) %>%
      torch::torch_stack()
  )
  list(known = known,
       observed = observed,
       static = static,
       target = target)
}

df_to_tensor <- function(df) {
  df %>%
    dplyr::mutate(dplyr::across(where(is.factor), as.integer)) %>%
    as.matrix() %>%
    torch::torch_tensor()
}




