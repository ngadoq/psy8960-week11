# Script Settings and Resources
library(tidyverse)
library(haven)
library(foreach)
library(caret)
library(parallel)
library(doParallel)

# Data Import and Cleaning
gss_import_tbl <- read_sav("GSS2016.sav") %>%
  filter(!is.na(MOSTHRS)) %>% 
  select(-c(HRS1, HRS2))
gss_tbl <- 
  gss_import_tbl[, colMeans(is.na(gss_import_tbl)) < .75] %>%
  rename(workhours = MOSTHRS) %>%
  mutate(workhours = as.integer(workhours))

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

# Modify number of cores to run on MSI
local_cluster <- makeCluster(64)
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
# Turn off parallel processing
stopCluster(local_cluster)
registerDoSEQ()


resample_sum <- summary(resamples(list(model_ols, model_glmnet, model_rf, model_xgb)))


# Publication
table3_tbl <- tibble(
  algo = c("lm","glmnet","ranger","xgbTree"),
  cv_rsq = str_remove(round(
    resample_sum$statistics$Rsquared[,"Mean"],2
  ),"^0"),
  ho_rsq = str_remove(c(
    format(round(hocv_cor_ols,2),nsmall=2),
    format(round(hocv_cor_glmnet,2),nsmall=2),
    format(round(hocv_cor_rf,2),nsmall=2),
    format(round(hocv_cor_xgb,2),nsmall=2)
  ),"^0")
) %>% 
  write_csv("table3.csv")

table3_tbl

# Running time
table4_tbl <- data.frame(
  algo = c("lm","glmnet","ranger","xgbTree"),
  supercomputer = c(time_non_parallel_ols[3], time_non_parallel_glmnet[3], time_non_parallel_rf[3], time_non_parallel_xgb[3]),
  supercomputer_64 = c(time_parallel_ols[3], time_parallel_glmnet[3], time_parallel_rf[3], time_parallel_xgb[3])
) %>% 
  write_csv("table4.csv")

table4_tbl

# XGBoost benefited most from moving to the supercomputer. When comparing execution time for parallel models between supercomputer and local computer, running time for XGBoost decreased significantly (almost 95%), followed by random forest (88%). Due to the complexity of these models, having more CPU definitely will benefit with the execution time.
# Running time decreases with the number of cores used for all models except OLS regression. This is especially true for complex models like random forest and XGBoost (having 64 cores in the MSI speeds up the process compared to 7 cores in local computer)

# I would recommend using the supercomputer because it helps data analysis process becomes much more efficient compared to using local computer. We can run complex models such as random forest and XGBoost within seconds and with the same set up. 
