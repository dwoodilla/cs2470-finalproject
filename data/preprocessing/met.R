options("menu.graphics" = FALSE)

install.packages("worldmet")
install.packages("tidyverse")
install.packages("jsonlite")
library(worldmet)
library(tidyverse)
library(jsonlite)

years = 2014:2025

df_list = map(years, ~ importNOAA(code = "997278-99999", year = .x))

ds = reduce(purrr::compact(df_list), full_join)

write_json(ds, path="./data/downloaded/met_ds.json")
