titanic_train <- read.csv(file = "~/Documents/kaggle/titanic/train.csv")
titanic_test <- read.csv(file = "~/Documents/kaggle/titanic/test.csv")
titanic_gender_submission <- read.csv(file = "~/Documents/kaggle/titanic/gender_submission.csv")

titanic_train_mod <- titanic_train %>% mutate(AgeBin = ifelse(is.na(Age), -0.5, Age)) %>%
  mutate(AgeBin = cut(AgeBin, 
                      breaks = c(-1, 0, 5, 12, 18, 35, 60, 100), 
                      labels = c("Missing", "Infant", "Child", "Teenage", "Young Adult", "Adult", "Senior"),
                      ordered_result = TRUE)) %>%
  dummy_cols(c("Pclass", "Sex", "AgeBin"))

colnames(titanic_train_mod)[23] <- "AgeBin_Young_Adult"

covars <- colnames(titanic_train_mod)[14:25]

task <- sl3_Task$new(data = titanic_train_mod,
                     covariates = covars,
                     outcome = "Survived")

slscreener <- Lrnr_pkg_SuperLearner_screener$new("screen.glmnet")
glm_learner <- Lrnr_glm$new()
screen_and_glm <- Pipeline$new(slscreener,
                               glm_learner)

SL.glmnet_learner <- Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.glmnet")

lrn_glm <- Lrnr_glm$new()
lrn_mean <- Lrnr_mean$new()
# penalized regressions:
lrn_ridge <- Lrnr_glmnet$new(alpha = 0)
lrn_lasso <- Lrnr_glmnet$new(alpha = 1)
# spline regressions:
lrn_polspline <- Lrnr_polspline$new()
lrn_earth <- Lrnr_earth$new()

# fast highly adaptive lasso (HAL) implementation
lrn_hal <- Lrnr_hal9001$new(max_degree = 2, num_knots = c(3,2), nfolds = 5)

# tree-based methods
lrn_ranger <- Lrnr_ranger$new()
lrn_xgb <- Lrnr_xgboost$new()

lrn_gam <- Lrnr_gam$new()
lrn_bayesglm <- Lrnr_bayesglm$new()

learner_stack <- Stack$new(
  lrn_glm, lrn_mean, lrn_ridge, lrn_lasso, lrn_polspline, lrn_earth, lrn_hal, 
  lrn_ranger, lrn_xgb, lrn_gam, lrn_bayesglm
)

sl <- Lrnr_sl$new(learners = learner_stack, metalearner = Lrnr_nnls$new())

stack_fit <- sl$train(task)

preds <- stack_fit$predict()

titanic_test_mod <- titanic_test %>% mutate(AgeBin = ifelse(is.na(Age), -0.5, Age)) %>%
  mutate(AgeBin = cut(AgeBin, 
                      breaks = c(-1, 0, 5, 12, 18, 35, 60, 100), 
                      labels = c("Missing", "Infant", "Child", "Teenage", "Young Adult", "Adult", "Senior"),
                      ordered_result = TRUE)) %>%
  dummy_cols(c("Pclass", "Sex", "AgeBin"))

colnames(titanic_test_mod)[22] <- "AgeBin_Young_Adult"

prediction_task <- sl3_Task$new(data = titanic_test_mod,
                                covariates = covars)

sl_preds_test <- stack_fit$predict(task = prediction_task)

hal_train <- fit_hal(X = titanic_train_mod[,14:25], Y = titanic_train_mod$Survived, smoothness_orders = 0, 
                     max_degree = 1, family = "binomial", fit_control = list(nfolds = 5), num_knots = 10)
hal_preds <- predict(hal_train, titanic_test_mod[13:24])
plot(hal_preds, titanic_gender_submission$Survived)
