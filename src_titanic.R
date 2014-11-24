# 1, dependencies

## A, packages
library(plyr)
library(dplyr)
library(DescTools)
library(stringr)
library(caret)
library(doMC)
library(caret)
library(mlbench)
library(caretEnsemble)
library(caTools)
library(rpart)

# 2, functions & controls

## A, import/prepare data for processing

train <- 'data/train.csv'
test <- 'data/test.csv'

prepareData <- function(train, test) {
  x <- read.csv(train, stringsAsFactors = F) 
  names(x) <- toupper(names(x) )
  x$SET <- "train"
  
  y <- read.csv(test, stringsAsFactors = F) 
  names(y) <- toupper(names(y) )
  y$SET <- "test"
  
  SURVIVED <- factor(x$SURVIVED) 
  levels(SURVIVED) <- list("no" = 0, "yes" = 1)
  
  common.names <- intersect(colnames(x), colnames(y) )
  x <- x[, common.names]
  y <- y[, common.names]
  z <- rbind_list(x, y)
  
  ### extract title - extracts title (e.x., Mr, Mrs, Ms, etc.)
  z.title <- sapply(z$NAME, FUN = function(x) {
    strsplit(x, split='[,.]')[[1]][2]
    } ) 
  z.title <- str_trim(z.title)
  z$TITLE <- z.title 
  
  ### extract surname - extracts person's surname
  z.surname <- sapply(z$NAME, FUN = function(x) {
    strsplit(x, split='[,.]')[[1]][1]
    } )
  z.surname <- str_trim(z.surname)
  z$SURNAME <- z.surname
  
  ### extract median fare
  median.fare <- z %>%
  group_by(PCLASS) %>%
  summarise(M.FARE = median(FARE, na.rm = T))
  
  ### data munging: binary-family, impute-embarked, category-title
  z <- z %>%
    mutate(TITLE = ifelse(TITLE == "Mr", 
                          1,
                   ifelse(TITLE == "Mrs" |
                          TITLE == "Mme", 
                          2,
                   ifelse(TITLE == "Master", 
                          3,
                   ifelse(TITLE == "Miss" | 
                          TITLE == "Ms" | 
                          TITLE == "Mlle", 
                          4,
                   ifelse(TITLE == "Col" | 
                          TITLE == "Major" | 
                          TITLE == "Capt", 
                          5,
                   ifelse(TITLE == "Dona" | 
                          TITLE == "Lady" |
                          TITLE == "the Countess",
                          6, 
                   ifelse(TITLE == "Rev",
                          7,
                   ifelse(TITLE == "Dr",
                          8, 
                   ifelse(TITLE == "Don" | 
                          TITLE == "Sir" |
                          TITLE == "Jonkheer",
                          9,                           
                          NA))))))))),
           FARE = ifelse(PCLASS == 1 & (is.na(FARE) | FARE == 0.0000), median.fare$M.FARE[median.fare$PCLASS == 1],
                  ifelse(PCLASS == 2 & (is.na(FARE) | FARE == 0.0000), median.fare$M.FARE[median.fare$PCLASS == 2],
                  ifelse(PCLASS == 3 & (is.na(FARE) | FARE == 0.0000), median.fare$M.FARE[median.fare$PCLASS == 3], 
                  FARE))) )

  z$LOG.FARE <- log(z$FARE)
  fit.age <- lm(AGE ~ PCLASS + TITLE + LOG.FARE + SEX + PCLASS:LOG.FARE-1,
                  data = z[!is.na(z$AGE),])
  fit.age <- step(fit.age, direction = "both", trace=0)
  z$AGE[is.na(z$AGE)] <- predict(fit.age, z[is.na(z$AGE),])
  
  z <- z %>%
    group_by(TICKET, SURNAME) %>%
    mutate(FAMILY = SIBSP + PARCH + 1,
           FAMILY = paste(as.character(FAMILY), SURNAME, sep = ""),
           EMBARKED = ifelse(EMBARKED == "", "C", EMBARKED) ) %>% 
    ungroup() %>% 
    group_by(FAMILY) %>%
    mutate(FAMILY3 = length(FAMILY),
           FAMILY2 = ifelse(FAMILY3 <= 2, "na.family", FAMILY),
           FAMILY.AMALE   = ifelse(SIBSP > 0 & PARCH > 0 & (TITLE == 1 | TITLE == 5 | TITLE == 9) & FAMILY3 > 2, 1, 0),
           FAMILY.AFEMALE = ifelse(SIBSP > 0 & PARCH > 0 & (TITLE == 2 | TITLE == 6) & FAMILY3 > 2, 1, 0),
           FAMILY.ACHILD  = ifelse(PARCH > 0 & (TITLE == 4 | TITLE == 3) & AGE < 18 & FAMILY3 > 2, 1, 0)
           ) %>%
        ungroup() %>%
    select(-NAME, -TICKET, -PASSENGERID, -CABIN, -FAMILY, -FAMILY3)
  
  z <- z %>%
    mutate(PCLASS   = factor(PCLASS),
           SEX      = factor(SEX),
           EMBARKED = factor(EMBARKED),
           TITLE    = factor(TITLE),
           FAMILY2   = factor(FAMILY2),
           SET   = factor(SET) )

  z_train <- filter(z, SET == "train") %>% 
  select(-SET, -LOG.FARE, -SURNAME)

  z_test <- filter(z, SET == "test") %>% 
  select(-SET, -LOG.FARE, -SURNAME)

  return(list(z_train, z_test, SURVIVED))
}

## B, training environment - grid search and model parameters
registerDoMC(cores = detectCores() - 1); getDoParWorkers()
parition.split <- 0.80
seed <- 50

num.folds   <- 10
num.repeats <- 5

train.ctrl <- trainControl(method = "repeatedcv",
                           number = num.folds,
                           repeats = num.repeats,
                           savePredictions = T,
                           verboseIter = F,
                           returnResamp = "all",
                           allowParallel = T,
                           classProbs = T,
                           summaryFunction = twoClassSummary)

gbmGrid <- expand.grid(n.trees = seq(100, 1000, by = 50),
                       interaction.depth = seq(1, 7, by = 2),
                       shrinkage = c(0.01, 0.1))

glmnetGrid <- expand.grid(.alpha = seq(0.1, 1.0, 0.1),
                          .lambda = seq(0.01, 0.5, 0.01))

rfGrid <- expand.grid(.mtry = seq(70, 80, 1) )

trellis.par.set(caretTheme())
PP <- c("center", "scale")

lazyModeling <- function(df, model.type, use.grid = TRUE, grid) {
      if (use.grid) 
        train(SURVIVED ~ ., 
              data = df,
              method = model.type, 
              tuneGrid = grid,
              preProcess = PP,
              metric = 'ROC', 
              trControl = train.ctrl)
      else
        train(SURVIVED ~ ., 
              data = df,
              method = model.type, 
              tuneLength = 10,
              preProcess = PP,
              metric = 'ROC', 
              trControl = train.ctrl)
}

# 3, procedure

## A, prepare data
train.df <- data.frame(prepareData(train, test)[1])
test.df  <- data.frame(prepareData(train, test)[2])

SURVIVED <- data.frame(prepareData(train, test)[3])
names(SURVIVED) <- "SURVIVED"
train.df <- cbind(SURVIVED, train.df)
rm(SURVIVED)
gc()

## B, describe data
Desc(train.df, plot = T)

## C, training
set.seed(seed)
random <- createDataPartition(train.df$SURVIVED, 
                              p = parition.split, list = F)
train.cv <- train.df[random, ]
train.predict <- train.df[-random, ]
rm(train.df)
rm(random)

## C.i., gbm
set.seed(seed)
gbmTune <- lazyModeling(train.cv, "gbm", use.grid = T, gbmGrid)

gbmImp <- varImp(gbmTune)
plot(gbmTune)
plot(gbmImp)
gc()

gbm.prediction <- predict(gbmTune, newdata = train.predict)
confusionMatrix(gbm.prediction, train.predict$SURVIVED)

## C.ii., svm
set.seed(seed)
svmTune <- lazyModeling(train.cv, "svmRadial", use.grid = F)
svmImp <- varImp(svmTune)
plot(svmTune)
plot(svmImp)
gc()

svm.prediction <- predict(svmTune, newdata = train.predict)
confusionMatrix(svm.prediction, train.predict$SURVIVED)
 
## C.iii., glmnet
set.seed(seed)
glmnetTune <- lazyModeling(train.cv, "glmnet", use.grid = T, glmnetGrid)
glmnetImp <- varImp(glmnetTune)
plot(glmnetTune)
plot(glmnetImp)
gc()

glmnet.prediction <- predict(svmTune, newdata = train.predict)
confusionMatrix(glmnet.prediction, train.predict$SURVIVED)

## C.iv., rf
set.seed(seed)
rfTune <- lazyModeling(train.cv, "rf", use.grid = T, rfGrid)
rfImp <- varImp(rfTune)
plot(rfTune)
plot(rfImp)
gc()

rf.prediction <- predict(rfTune, newdata = train.predict)
confusionMatrix(rf.prediction, train.predict$SURVIVED)

# 4, voting

## A, list tuning models
model.list <- list(glmnetTune, rfTune, svmTune, gbmTune)
modelCor(resamples(model.list))

model.pred <- lapply(model.list, predict, newdata = train.predict, type = "prob")
model.pred <- lapply(model.pred, function(x) x[, "yes"])
model.pred <- data.frame(model.pred)
names(model.pred) <- c("glmnetTune", "rfTune", "svmTune", "gbmTune")

colAUC(model.pred, train.predict$SURVIVED)

voting <- model.pred %>%
  mutate(VOTING = (glmnetTune*2 + rfTune + svmTune + gbmTune*2) / 6,
         VOTING = ifelse(VOTING >= 0.5, "yes", "no") )

confusionMatrix(train.predict$SURVIVED, voting$VOTING)

# 5, build submission file
test.pred <- lapply(model.list, predict, newdata = test.df, type = "prob")
test.pred <- lapply(test.pred, function(x) x[, "yes"])
test.pred <- data.frame(test.pred)
names(test.pred) <- c("glmnetTune", "rfTune", "svmTune", "gbmTune")

voting <- test.pred %>%
  mutate(VOTING = (glmnetTune*2 + rfTune + svmTune + gbmTune*2) / 6,
         VOTING = ifelse(VOTING >= 0.5, 1, 0) )

test.submit <- read.csv("data/test.csv", stringsAsFactors = F)

PassengerId <- select(test.submit, PassengerId)
test.submit <- cbind(PassengerId, voting$VOTING)
names(test.submit) <- c("PassengerId", "Survived")

write.csv(test.submit, "submission20.csv", row.names = FALSE)
