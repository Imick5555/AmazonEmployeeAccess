library(tidymodels)
library(tidyverse)
library(embed)
library(vroom)
library(recipes)
library(embed)
library(workflows)
library(rpart)
library(ranger)
library(dials)
library(themis)

train <- vroom("train.csv")
test <- vroom("test.csv")

# DataExplorer::plot_intro(train)
# DataExplorer::plot_correlation(train)
# DataExplorer::plot_bar(train)

train <- train %>% mutate(ACTION = factor(ACTION))
my_recipe <- recipe(ACTION ~., data=train) %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .005) %>% # combines categorical values that occur <.1% i
  #step_dummy(all_nominal_predictors()) %>%  # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())
# apply the recipe to your data

prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet13
bake(prepped_recipe, new_data=train)

# logRegModel <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model3
#   set_engine("glmnet")

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees=1000) %>%
set_engine("ranger") %>%
set_mode("classification")

amazon_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(my_mod)

# param_set <- extract_parameter_set_dials(amazon_workflow)
# param_set <- finalize(param_set, train)
# 
# tuning_grid <- grid_regular(param_set, levels = 2)


folds <- vfold_cv(train, v = 5, repeats=1)

tuning_grid <- grid_regular(mtry(range = c(2, 9)),
                            min_n(range = c(1, 8)),
                            levels = 3)
## Run the CV18
CV_results <- amazon_workflow %>%
      tune_grid(resamples=folds,
                grid=tuning_grid,
                metrics=metric_set(accuracy))

best_tune <- CV_results |>
  select_best(metric="accuracy")

final_wf <- amazon_workflow |>
  finalize_workflow(best_tune)|>
  fit(data = trainData)



## Make predictions8
ggg_predictions <- final_wf %>%
                            predict(new_data=testData,
                              type="prob") %>%
                              bind_cols(test) %>%
                              rename(ACTION=.pred_1) %>%
                              select(id,ACTION)# "class" or "prob"
# NOTE: some of these step functions are not appropriate to use together


#kaggle_submission <- amazon_predictions %>%
 # bind_cols(test %>% select(MGR_ID)) %>%
  #select(MGR_ID, .pred_1) %>%
  #rename(ACTION=.pred_1) 
  # mutate(count=pmax(0, count)) %>%
  # mutate(count = exp(count)) %>%
  # mutate(datetime=as.character(format(datetime)))

# logRegPreds <- logReg_workflow %>%
#   predict(new_data=test, type="prob") %>%
#   bind_cols(test) %>%
#   rename(ACTION=.pred_1) %>%
#   select(id, ACTION)

vroom_write(x=amazon_predictions, file="./AmazonPreds.csv", delim=",")
#vroom::vroom_write(x = kaggle_submission, file = "AmazonPreds.csv")



