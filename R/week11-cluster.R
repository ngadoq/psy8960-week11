# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
library(tictoc)
library(parallel)
library(doParallel)

# Data Import and Cleaning
gss_import_tbl <- read_sav("../data/GSS2016.sav") %>%
  filter(!is.na(MOSTHRS)) %>% 
  select(-c(HRS1, HRS2))
gss_tbl <- 
  gss_import_tbl[, colMeans(is.na(gss_import_tbl)) < .75] %>%
  rename(workhours = MOSTHRS) %>%
  mutate(workhours = as.integer(workhours))


# Visualization
ggplot(gss_tbl, aes(x=workhours)) + geom_histogram()

# Analysis
train_cases <- sample(1:nrow(gss_tbl), .75*nrow(gss_tbl))

gss_train_tbl <- gss_tbl[train_cases, ]
gss_test_tbl <- gss_tbl[-train_cases, ]

training_folds <- createFolds(gss_train_tbl$workhours,
                              k=10)


time_non_parallel_ols <- system.time({
  model_ols <- train(
    workhours ~ .,
    gss_train_tbl, 
    method="lm",
    na.action=na.pass,
    preProcess=c("center", "scale", "nzv", "medianImpute"),
    trControl=trainControl(method="cv", number=10, indexOut=training_folds) 
  )
}
)


hocv_cor_ols <- cor(
  predict(model_ols, gss_test_tbl, na.action=na.pass),
  gss_test_tbl$workhours
) ^ 2

time_non_parallel_glmnet <- system.time({
  model_glmnet <- train(
    workhours ~ .,
    gss_train_tbl, 
    method="glmnet",
    tuneLength=3,
    na.action=na.pass,
    preProcess=c("center", "scale", "nzv", "medianImpute"),
    trControl=trainControl(method="cv", number=10, indexOut=training_folds) 
  )
}
)


hocv_cor_glmnet <- cor(
  predict(model_glmnet, gss_test_tbl, na.action=na.pass),
  gss_test_tbl$workhours
) ^ 2

time_non_parallel_rf <- system.time({
  model_rf <- train(
    workhours ~ .,
    gss_train_tbl, 
    method="ranger",
    tuneLength=3,
    na.action=na.pass,
    preProcess=c("center", "scale", "nzv", "medianImpute"),
    trControl=trainControl(method="cv", number=10, indexOut=training_folds, verboseIter=T) 
  )
}
)

hocv_cor_rf <- cor(
  predict(model_rf, gss_test_tbl, na.action=na.pass),
  gss_test_tbl$workhours
) ^ 2

time_non_parallel_xgb <- system.time({
  model_xgb <- train(
    workhours ~ .,
    gss_train_tbl, 
    method="xgbTree",
    tuneLength=3,
    na.action=na.pass,
    preProcess=c("center", "scale", "nzv", "medianImpute"),
    trControl=trainControl(method="cv", number=10, indexOut=training_folds, verboseIter=T) 
  )
}
)


hocv_cor_xgb <- cor(
  predict(model_xgb, gss_test_tbl, na.action=na.pass),
  gss_test_tbl$workhours
) ^ 2


local_cluster <- makeCluster(7)
registerDoParallel(local_cluster)

time_parallel_ols <- system.time({
  model_ols <- train(
    workhours ~ .,
    gss_train_tbl, 
    method="lm",
    na.action=na.pass,
    preProcess=c("center", "scale", "nzv", "medianImpute"),
    trControl=trainControl(method="cv", number=10, indexOut=training_folds) 
  )
}
)

hocv_cor_ols <- cor(
  predict(model_ols, gss_test_tbl, na.action=na.pass),
  gss_test_tbl$workhours
) ^ 2

time_parallel_glmnet <- system.time({
  model_glmnet <- train(
    workhours ~ .,
    gss_train_tbl, 
    method="glmnet",
    tuneLength=3,
    na.action=na.pass,
    preProcess=c("center", "scale", "nzv", "medianImpute"),
    trControl=trainControl(method="cv", number=10, indexOut=training_folds) 
  )
}
)

hocv_cor_glmnet <- cor(
  predict(model_glmnet, gss_test_tbl, na.action=na.pass),
  gss_test_tbl$workhours
) ^ 2

time_parallel_rf <- system.time({
  model_rf <- train(
    workhours ~ .,
    gss_train_tbl, 
    method="ranger",
    tuneLength=3,
    na.action=na.pass,
    preProcess=c("center", "scale", "nzv", "medianImpute"),
    trControl=trainControl(method="cv", number=10, indexOut=training_folds, verboseIter=T) 
  )
}
)

hocv_cor_rf <- cor(
  predict(model_rf, gss_test_tbl, na.action=na.pass),
  gss_test_tbl$workhours
) ^ 2

time_parallel_xgb <- system.time({
  model_xgb <- train(
    workhours ~ .,
    gss_train_tbl, 
    method="xgbTree",
    tuneLength=3,
    na.action=na.pass,
    preProcess=c("center", "scale", "nzv", "medianImpute"),
    trControl=trainControl(method="cv", number=10, indexOut=training_folds, verboseIter=T) 
  )
}
)

resample_sum <- summary(resamples(list(model_ols, model_glmnet, model_rf, model_xgb)))


# Publication