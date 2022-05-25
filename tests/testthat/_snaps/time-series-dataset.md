# can create a spec

    Code
      print(spec)
    Output
      A <tft_dataset_spec> with:
      
      x `lookback` and `horizon` are not set. Use `spec_time_splits()` 
      
      -- Covariates: 
      x `index` is not set. Use `spec_covariate_index()` to set it.
      x `keys` is not set. Use `spec_covariate_keys()` to set it.
      ! `static` is not set. Use `spec_covariate_static()` to set it.
      ! `known` is not set. Use `spec_covariate_known()` to set it.
      ! `unknown` is not set. Covariates that are not listed as other types are considered `unknown`.
      
      i Call `prep()` to prepare the specification.

---

    A <tft_dataset_spec> with:
    
    v lookback = 52 and horizon = 4.
    
    -- Covariates: 
    v `index`: Date
    v `keys`: <list: Store, Dept>
    v `static`: <list: Type, Size>
    v `known`: <list: starts_with("MarkDown"), starts_with("Date_"), starts_with("na_ind")>
    ! `unknown` is not set. Covariates that are not listed as other types are considered `unknown`.
    
    i Call `prep()` to prepare the specification.

---

    A <prepared_tft_dataset_spec> with:
    
    v lookback = 52 and horizon = 4.
    v The number of possible slices is 320
    
    -- Covariates: 
    v `index`: Date
    v `keys`: Store and Dept
    v `static`: Type and Size
    v `known`: MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5, na_ind_MarkDown1, na_ind_MarkDown2, na_ind_MarkDown3, na_ind_MarkDown4, and na_ind_MarkDown5
    v `unknown`: IsHoliday, Temperature, Fuel_Price, CPI, Unemployment, and intercept
    i Variables that are not specified in other types are considered `unknown`.
    
    i Call `transform()` to apply this spec to a different dataset.

