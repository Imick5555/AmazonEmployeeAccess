library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(recipes)
library(embed)
library(workflows)
library(rpart)
library(ranger)
library(discrim)
library(naivebayes)
library(remotes)
library(dplyr)
library(reticulate)
library(dials)

train <- vroom("train.csv")
test <- vroom("test.csv")

train <- train %>% mutate(ACTION = factor(ACTION))
my_recipe <- recipe(ACTION ~., data=train) %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  update_role(MGR_ID, new_role="id") %>%
  #step_mutate(color = factor(color)) %>%
  #step_dummy(color, one_hot = TRUE) %>%
  step_range(all_numeric_predictors(),min=0,max=1) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  step_pca(all_predictors(), threshold=0.5)
#step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
# apply the recipe to your data

prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet13
bake(prepped_recipe, new_data=train)

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees=500) %>%
set_engine("ranger") %>%
set_mode("classification")

pca_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(my_mod)


folds <- vfold_cv(train, v = 2, repeats=1)

# mtry_vals <- finalize(mtry(), train %>% select(-ACTION))
# 
# tune_grid <- grid_regular(
#   parameters(
#     mtry_vals,                      
#     min_n(range = c(2, 10))         
#   ),
#   levels = 2                         
# )

tuned_pca  <- pca_workflow |>
  tune_grid(
    resamples = folds,
    grid = tune_grid,
    metrics = metric_set(roc_auc, accuracy)
  )

CV_results <- pca_workflow %>%
  tune_grid(resamples=folds,
            metrics=metric_set(roc_auc, accuracy))

bestTune <- CV_results %>%
  select_best(metric="accuracy")

final_wf <-
  pca_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_wf %>%
  predict(new_data = test)

pca_predictions <- predict(final_wf, new_data = test)



pca_predictions


kaggle_submission <- final_wf %>%
  predict(new_data=test,
          type="prob") %>%
  bind_cols(test) %>%
  rename(ACTION=.pred_1) %>%
  select(id,ACTION)

vroom_write(x=kaggle_submission, file="./AmazonPreds.csv", delim=",")
