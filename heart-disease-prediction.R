library(data.table)
library(tidyverse)

# 1. : displays the age of the individual.
# 2. : displays the gender of the individual using the following format : 1 = male 0 =
#   female
# 3. : displays the type of chest-pain experienced by the individual using the following
# format : 1 = typical angina 2 = atypical angina 3 = non - anginal pain 4 =
#   asymptotic
# 4. : displays the resting blood pressure value of an individual in mmHg (unit)
# 5. : displays the serum cholesterol in mg/dl (unit)
# 6. : compares the fasting blood sugar value of an individual with 120mg/dl. If fasting
# blood sugar > 120mg/dl then : 1 (true) else : 0 (false)
# 7. : displays resting electrocardiographic results 0 = normal 1 = having ST-T wave
# abnormality 2 = left ventricular hyperthrophy
# 8. : displays the max heart rate achieved by an individual.
# 9. : 1 = yes 0 = no
# 10. : displays the value which is an integer or float.
# 11. : 1 = upsloping 2 = flat 3 = downsloping
# 12. : displays the value as integer or float.
# 13. : displays the thalassemia : 3 = normal 6 = fixed defect 7 = reversible defect
# 14. : Displays whether the individual is suffering from heart disease or not : 0 =
#   absence 1, 2, 3, 4 = present.

heartDataFrame=fread("heart-disease.csv",header=T,check.names=F,data.table = F)
str(heartDataFrame)

heartDataFrame[,12:13] <- sapply(heartDataFrame[,12:13],as.numeric)
heartDataFrame=na.omit(heartDataFrame)

unlist(lapply(heartDataFrame, function(x) {sum(is.na(x)) }))

corMatrix=cor(heartDataFrame)
library(corrplot)
corrplot(corMatrix, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

xtabs(formula = "~predicted",data = heartDataFrame)
# ggplot(heartDataFrame, aes(x= predicted, fill=predicted)) + 
#   geom_bar() +
#   xlab("Heart Disease") +
#   ylab("Count") +
#   ggtitle("Analysis of Presence and Absence of Heart Disease") +
#   scale_fill_discrete(name = "Heart Disease"))


boxplot(age~predicted,data = heartDataFrame)

# -- Value 1: typical angina
# -- Value 2: atypical angina
# -- Value 3: non-anginal pain
# -- Value 4: asymptomatic

data2 <- heartDataFrame %>% 
  mutate(sex = if_else(sex == 1, "MALE", "FEMALE"),
         fbs = if_else(fbs == 1, ">120", "<=120"),
         exang = if_else(exang == 1, "YES" ,"NO"),
         cp = factor(cp,labels = c("TYPICAL ANGINA","ATYPICAL ANGINA","NON-ANGINAL PAIN", "ASYMPTOMATIC")),
         restecg = if_else(restecg == 0, "NORMAL",
                           if_else(restecg == 1, "ABNORMALITY", "PROBABLE OR DEFINITE")),
         slope = factor(slope,labels =c("upsloping","flat","downsloping")),
         ca = as.factor(ca),
         thal = factor(thal,labels =c("normal","fixed defect","reversible defect")),
         predicted = if_else(predicted == 0, "NO", "YES")
  ) %>% 
  mutate_if(is.character, as.factor) %>% 
  select(predicted, sex, fbs, exang, cp, restecg, slope, ca, thal, everything())


ggplot(data2, aes(cp, fill = predicted))+
  geom_bar(position = position_dodge(width=0.25),width = 0.2)+
  ggtitle("cp") + theme_minimal()
# ASYMPTOMATIC

ggplot(data2, aes(restecg, fill = predicted))+
  geom_bar(position = position_dodge(width=0.25),width = 0.2)+
  ggtitle("restecg")
#  PROBABLE OR DEFINITE, NORMAL

ggplot(data2, aes(slope, fill = predicted))+
  geom_bar(position = position_dodge(width=0.25),width = 0.2)+
  ggtitle("slope")
# flat slope

ggplot(data2, aes(thal, fill = predicted))+
  geom_bar(position = position_dodge(width=0.25),width = 0.2)+
  ggtitle("thal")
# reversible defect

ggplot(data2, aes(fbs, fill = predicted))+
  geom_bar(position = position_dodge(width=0.25),width = 0.2)+
  ggtitle("fbs") + theme_minimal()
# low fbs


# Model data----------

library(pROC)
library(caret)

train_index <- createDataPartition(data2$predicted, p = 0.7, list = FALSE, times = 1)


train <- data2[train_index,]
test <- data2[-train_index,]

train_x <- train %>% dplyr::select(-predicted)
train_y <- train$predicted

test_x <- test %>% dplyr::select(-predicted)
test_y <- test$predicted


training <- data.frame(train_x, predicted = train_y)

# GLM ----------

model_glm <- glm(predicted ~ ., 
                 data = training, 
                 family = "binomial")
model_glm

summary(model_glm)
test_ol <- predict(model_glm, newdata = test_x, type = "response")

# GLM cross-validation ----------
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE )
glm_tune <- train(train_x, 
                  train_y, 
                  method = "glm",
                  trControl = ctrl)

# Train
confusionMatrix(data = predict(glm_tune, train_x),
                reference = train_y, positive = "YES")
# Test
confusionMatrix(data = predict(glm_tune, test_x),
                reference = test_y, positive = "YES")

roc(glm_tune$pred$obs, 
    glm_tune$pred$YES, 
    levels = rev(levels(glm_tune$pred$obs)),
    plot = TRUE, print.auc = TRUE)
summary(glm_tune$finalModel)


exp(coef(glm_tune$finalModel))

## odds ratios and 95% CI
exp(cbind(OR = coef(glm_tune$finalModel), confint(glm_tune$finalModel)))

# RF model ---------

library(randomForest)
rf_fit <- randomForest(train_x, train_y, importance = TRUE)

rf_fit

importance(rf_fit)

varImpPlot(rf_fit)

confusionMatrix(predict(rf_fit, test_x), test_y, positive = "YES")


# RF Model Tuning ------------
control <- trainControl(method='cv', 
                        number=10, 
                        search='grid')
# 10-folds, try from 1 to 10 as variables split 
tunegrid <- expand.grid(mtry = (1:10)) 

# to identify number of variables for split
rf_gridsearch <- train(predicted ~ ., 
                       data = train,
                       method = 'rf',
                       metric = 'Accuracy',
                       tuneGrid = tunegrid)


rf_gridsearch

rf_cm <- confusionMatrix(predict(rf_gridsearch, test_x), test_y, positive = "YES")
rf_cm

# Accuracy comparision -------
ds_rf <- defaultSummary(data.frame(obs = test_y, 
                                   pred = predict(rf_gridsearch, test_x)))
ds_rf

ds_logReg <- defaultSummary(data.frame(obs = test_y, 
                                   pred = predict(glm_tune, test_x)))
ds_logReg

comp <- data.frame(row.names = NULL,
                   Model = c("Random Forest", "Logistic Regression"),
                   Accuracy = c(ds_rf[1], ds_logReg[1])
                  ) %>% arrange(-Accuracy)
comp

# SVM --------------

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
svm_Linear <- train(predicted ~., data = train, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

test_pred <- predict(svm_Linear, newdata = test)
test_pred

confusionMatrix(table(test_pred, test_y))

grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
svm_Linear_Grid <- train(predicted ~., data = train, method = "svmLinear",
                         trControl=trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)
svm_Linear_Grid
plot(svm_Linear_Grid)

test_pred_grid <- predict(svm_Linear_Grid, newdata = test)
test_pred_grid

confusionMatrix(table(test_pred_grid, test_y))


# GBM --------

objControl <- trainControl(method='cv', number=10)

gbmGrid <-  expand.grid(interaction.depth =  c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode =10)
# run model
boostModel <- train(predicted ~ .,data=train, method='gbm',
                    trControl=objControl, tuneGrid = gbmGrid, verbose=F)

#plot(boostModel)
boostPrediction <- predict(boostModel, test)
boostPredictionprob <- predict(boostModel, test, type='prob')[2]

boostConfMat <- confusionMatrix(boostPrediction, test_y)

#ROC Curve
AUC = list()
Accuracy = list()
AUC$boost <- roc(as.numeric(test_y),as.numeric(as.matrix((boostPredictionprob))))$auc
Accuracy$boost <- boostConfMat$overall['Accuracy']  



# ANN -------


library(neuralnet)

cols = names(train)
nnFormula <- paste(cols[-c(1)], collapse = " + ", sep = "")
nnFormula <- paste("predicted", "~", nnFormula)
nnFormula <- as.formula(nnFormula)

trainNueralNet = heartDataFrame[train_index,]

nn <-
  neuralnet(
    nnFormula,
    data = trainNueralNet,
    hidden = c(2, 1),
    linear.output = F,
    threshold = 0.01
  )
nn$result.matrix
plot(nn)


#Test the resulting output
testNueralNet <- heartDataFrame[-train_index, -14]
nn.results <- compute(nn, testNueralNet)
tempDataFrame <- heartDataFrame[-train_index, ]
tempDataFrame[(which(tempDataFrame$predicted != 0)), "predicted"] = 1

RMSE.NN <-
  (
    sum((tempDataFrame$predicted - nn.results$net.result) ^ 2)
   / nrow(tempDataFrame)
  ) ^ 0.5
RMSE.NN
