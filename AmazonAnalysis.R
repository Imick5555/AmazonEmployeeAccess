library(tidymodels)
library(embed) # for target encoding

my_recipe <- recipe(rFormula, data=trainData) %>%
step_mutate_at(vars_I_want_to_mutate, fn = factor) %>% # turn all numeric features into factors
step_other(vars_I_want_other_cat_in, threshold = .001) %>% # combines categorical values that occur <.1% i
step_dummy(vars_I_want_to_dummy) %>% # dummy variable encoding
step_lencode_mixed(vars_I_want_to_target_encode, outcome = vars(target_var)) #target encoding (must be 2-f
# also step_lencode_glm() and step_lencode_bayes()


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = data_I_want_to_clean)
