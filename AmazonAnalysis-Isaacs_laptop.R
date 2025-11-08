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

remotes::install_github("rstudio/tensorflow")

reticulate::install_python()

keras::install_keras()
  
train <- vroom("train.csv")
test <- vroom("test.csv")

# DataExplorer::plot_intro(train)
# DataExplorer::plot_correlation(train)
# DataExplorer::plot_bar(train)

train <- train %>% mutate(ACTION = factor(ACTION))
my_recipe <- recipe(ACTION ~., data=train) %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  update_role(MGR_ID, new_role="id") %>%
  #step_mutate(color = factor(color)) %>%
  #step_dummy(color, one_hot = TRUE) %>%
  step_range(all_numeric_predictors(),min=0,max=1) %>%
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_dummy(all_nominal_predictors())  # dummy variable encoding
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
# apply the recipe to your data

prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet13
bake(prepped_recipe, new_data=train)

# logRegModel <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model3
#   set_engine("glmnet")

# my_mod <- rand_forest(mtry = tune(),
#                       min_n = tune(),
#                       trees=500) %>%
# set_engine("ranger") %>%
# set_mode("classification")

# nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
#     set_mode("classification") %>%
#     set_engine("naivebayes") 
nn_model <- mlp(hidden_units = tune(),
                epochs = 50 #or 100 or 2507
) %>%
set_engine("keras") %>% #verbose = 0 prints off less9
  set_mode("classification")

nn_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(nn_model)

maxHiddenUnits <- round((ncol(train) + length(levels(train$ACTION))) / 2)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, maxHiddenUnits)),
                            levels=2)

folds <- vfold_cv(train, v = 2, repeats=1)

tuned_nn  <- nn_workflow |>
  tune_grid(
    resamples = folds,
    grid = nn_tuneGrid,
    metrics = metric_set(roc_auc, accuracy)
  )

tuned_nn %>% collect_metrics() %>%
filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()



# param_set <- extract_parameter_set_dials(nb_workflow)
# param_set <- finalize(param_set, train)
# 
# tuning_grid <- grid_regular(param_set, levels = 2)
# 
# tuning_grid <- grid_regular(Laplace(),
#                             smoothness(),
#                             levels = 2)

#folds <- vfold_cv(train, v = 2, repeats=1)

## Run the CV18
CV_results <- nn_workflow %>%
      tune_grid(resamples=folds,
                grid=tuning_grid,
                metrics=metric_set(roc_auc, accuracy))

best_tune <- CV_results |>
  select_best(metric="roc_auc")

final_wf <- nn_workflow |>
  finalize_workflow(best_tune)|>
  fit(data = train)

## Make predictions8
amazon_predictions <- final_wf %>%
                            predict(new_data=test,
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



