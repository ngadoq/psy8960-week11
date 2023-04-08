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
  # Remove other two “work hours” variables 
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

# Non-parallel 
# Time to run non-parallel OLS model
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

# Time to run non-parallel elastic net model
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

# Time to run non-parallel random forest model
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

# Time to run non-parallel XGBoost model
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


# Parellization
# Turn on parallel processing
local_cluster <- makeCluster(7)
registerDoParallel(local_cluster)

# Time to run non-parallel OLS model
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

# Time to run non-parallel elastic net model
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

# Time to run non-parallel random forest model
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

# Time to run non-parallel XGBoost model
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
table1_tbl <- tibble(
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
) 

table1_tbl

# Running time
table2_tbl <- data.frame(
  algo = c("lm","glmnet","ranger","xgbTree"),
  original = c(time_non_parallel_ols[3], time_non_parallel_glmnet[3], time_non_parallel_rf[3], time_non_parallel_xgb[3]),
  parallelized = c(time_parallel_ols[3], time_parallel_glmnet[3], time_parallel_rf[3], time_parallel_xgb[3])
)
table2_tbl


