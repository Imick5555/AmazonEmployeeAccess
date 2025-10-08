library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(DataExplorer)
library(recipes)
library(embed)

setwd("C://Users//Isaac//OneDrive//Documents//fall 2025 semester//STAT 348//AmazonEmployeeAccess")

train <- vroom("train.csv")
test <- vroom("test.csv")

# DataExplorer::plot_intro(train)
# DataExplorer::plot_correlation(train)
# DataExplorer::plot_bar(train)

my_recipe <- recipe(ACTION ~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet13
bake(prepped_recipe, new_data=train)


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = data_I_want_to_clean)
