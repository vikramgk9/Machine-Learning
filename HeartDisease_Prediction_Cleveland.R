rm(list = ls())

installIfAbsentAndLoad <- function(neededVector) {
  for(thepackage in neededVector) {
    if( ! require(thepackage, character.only = TRUE) )
    { install.packages(thepackage)}
    require(thepackage, character.only = TRUE)
  }
}
needed <- c('boot', 'dplyr', 'e1071', 'ROCR', 'rpart', 'rattle')
installIfAbsentAndLoad(needed)

heart_df <- read.csv("processed.cleveland.data.csv", header = F)

###Pre-Process the data for building models#####

names(heart_df) <- c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                      "thalach","exang", "oldpeak","slope", "ca", "thal", "y")

# Create a working data set
heart_df_working <- heart_df

# Because people with heart disease is classified as 1,2,3,4. Converting all numbers greater than 0 as 1 and all numbers equal to 0 as 2, to have binomial classification
heart_df_working$y[heart_df_working$y > 0] <- 1
heart_df_working$y[heart_df_working$y == 0] <- 2

str(heart_df_working)

# There ares some '?' in the values of thal and ca. Convert them into NA and then remove them
heart_df_working[heart_df_working == '?'] <- NA

index <- apply(heart_df_working, 2, function(x) any(is.na(x) | is.infinite(x)))
names(heart_df_working[index])

heart_df_working <- na.omit(heart_df_working)

str(heart_df_working)

#####Change from numeric to factors for some variables##############################3
heart_df_working$sex <- as.factor(heart_df_working$sex)
heart_df_working$cp <- as.factor(heart_df_working$cp)
heart_df_working$fbs <- as.factor(heart_df_working$fbs)
heart_df_working$restecg <- as.factor(heart_df_working$restecg)
heart_df_working$exang <- as.factor(heart_df_working$exang)
heart_df_working$slope <- as.factor(heart_df_working$slope)
heart_df_working$y <- as.factor(heart_df_working$y)

str(heart_df_working)

##Check if any NAs are still existing
index <- apply(heart_df_working, 2, function(x) any(is.na(x) | is.infinite(x)))
names(heart_df_working[index])


###############################################################################################################################
########################################### Logistic Regression ################################################################
###############################################################################################################################

glm.fit <- glm(y ~ ., data = heart_df_working, family='binomial')
summary(glm.fit)

set.seed(1)
# Divide into train and test set
n <- nrow(heart_df_working)
train.indices <- sample(1:n, .7 * nrow(heart_df_working))
test.indices <- setdiff(1:n, train.indices)
hearttrain.df <- heart_df_working[train.indices, ]
hearttest.df <- heart_df_working[test.indices, ]

#### Get the Power and Error rate by tuning the cost on the training data
for(i in seq(0.1, 0.9, by = 0.1)){
  glm.fit <- glm(y ~ ., data = hearttrain.df, family='binomial')
  glm.probs <- predict(glm.fit, hearttrain.df, type='response')
  glm.pred <- ifelse(glm.probs > i, '2', '1')
  mytable <- table(hearttrain.df$y, glm.pred)
  print(mytable)
  print(paste('Error rate of the model with Cost', i, 'is', mean(glm.pred != hearttrain.df$y)))
  print(paste('The power of the model with Cost',i, 'is', mytable['1', '1']/sum(mytable['1', ])))
}

# Model with cost = 0.6 seems to be the best model, when trying to achieve a balance between Power and Error Rate
# Calculate the 5 fold CV error

cost<-function(actuals, predictions) mean(abs(actuals-predictions) > 0.6)
cv.err <- cv.glm(hearttrain.df, glm(y ~ ., 
                                    data = hearttrain.df, 
                                    family ='binomial'), 
                 cost, K = 5)$delta[1]
cv.err

#Let's see the performance on the test set with varying costs

for(i in seq(0.1, 0.9, by = 0.1)){
  glm.fit <- glm(y ~ ., data = hearttest.df, family='binomial')
  glm.probs <- predict(glm.fit, hearttest.df, type='response')
  glm.df <- data.frame(x = glm.probs, y = hearttest.df$y)
  glm.pred <- ifelse(glm.probs > i, '2', '1')
  mytable <- table(hearttest.df$y, glm.pred)
  print(mytable)
  print(paste('Error rate of the model with Cost', i, 'is', mean(glm.pred != hearttest.df$y)))
  print(paste('The power of the model with Cost',i, 'is', mytable['1', '1']/sum(mytable['1', ])))
}

# Model with cost 0.6 seems to be a right balance between achieving fairly high power of the model and the overall error rate. Or in other words, cost at 0.6 seems to provide the right balance between variance and bias
# Therefore, choose the model with cost = 0.6 as the final model

###############################################################################################################################
################################################### SVM #######################################################################
###############################################################################################################################

############################################### Using Linear Kernel #####################################################
set.seed(1)
# Fit different SVM models with varying costs
svmfit.cost0.001 <- svm(y ~ ., data = hearttrain.df , kernel="linear", cost=0.001, scale = T)
svmfit.cost0.01 <- svm(y ~ ., data = hearttrain.df , kernel="linear", cost=0.01, scale = T)
svmfit.cost0.1 <- svm(y ~ ., data = hearttrain.df , kernel="linear", cost=0.1, scale = T)
svmfit.cost1 <- svm(y ~ ., data = hearttrain.df , kernel="linear", cost=1, scale = T)
svmfit.cost5 <- svm(y ~ ., data = hearttrain.df , kernel="linear", cost=5, scale = T)
svmfit.cost10 <- svm(y ~ ., data = hearttrain.df , kernel="linear", cost=10, scale = T)
svmfit.cost100 <- svm(y ~ ., data = hearttrain.df , kernel="linear", cost=100, scale = T)

# Let us see which of the models produce best power
for(i in c(0.001, 0.01, 0.1, 1,5, 10, 100)){
  svmfit <- svm(y ~ ., data = hearttrain.df , kernel="linear", cost=i, scale = T)
  ypred <- predict(svmfit, hearttrain.df)
  mytable <- table(actaul = hearttrain.df$y, predict = ypred)
  print(paste('The power of the model for Cost', i, 'is', mytable['1', '1']/sum(mytable['1', ])))
}

# Let us see the best model according to lowest CV error
tune.out <- tune(svm,y ~ ., data=hearttrain.df, kernel="linear", 
                 ranges=list(cost=c(0.001, 0.01, 0.1, 1,5, 10, 100)))

summary(tune.out)
bestmod <- tune.out$best.model
bestmod

##############################################Testing on the test set############################################################3

ypred.0.001 <- predict(svmfit.cost0.001, hearttest.df)
mytable <- table(actual=hearttest.df$y, predict=ypred.0.001)
mytable
print(paste('The power of the model for Cost 0.001 is', mytable['1', '1']/sum(mytable['1', ])))

ypred.0.01 <- predict(svmfit.cost0.01, hearttest.df)
mytable <- table(actual=hearttest.df$y, predict=ypred.0.01)
mytable
print(paste('The power of the model for Cost 0.01 is', mytable['1', '1']/sum(mytable['1', ])))

ypred.0.1 <- predict(svmfit.cost0.1, hearttest.df)
mytable <- table(actual=hearttest.df$y, predict=ypred.0.1)
mytable
print(paste('The power of the model for Cost 0.1 is', mytable['1', '1']/sum(mytable['1', ])))

ypred.1 <- predict(svmfit.cost1, hearttest.df)
mytable <- table(actual=hearttest.df$y, predict=ypred.1)
mytable
print(paste('The power of the model for Cost 1 is', mytable['1', '1']/sum(mytable['1', ])))

ypred.5 <- predict(svmfit.cost5, hearttest.df)
mytable <- table(actual=hearttest.df$y, predict=ypred.5)
mytable
print(paste('The power of the model for Cost 5 is', mytable['1', '1']/sum(mytable['1', ])))

ypred.10 <- predict(svmfit.cost10, hearttest.df)
mytable <- table(actual=hearttest.df$y, predict=ypred.10)
mytable
print(paste('The power of the model for Cost 10 is', mytable['1', '1']/sum(mytable['1', ])))

ypred.100 <- predict(svmfit.cost100, hearttest.df)
mytable <- table(actual=hearttest.df$y, predict=ypred.100)
mytable
print(paste('The power of the model for Cost 100 is', mytable['1', '1']/sum(mytable['1', ])))

# Conclusion: Best Model from Linear SVM is a model with Cost 100 and Gamma 0.04347826

################################### Using Radial Kernel ########################################################################
set.seed(1)
tune.out <- tune(svm, y~., data = hearttrain.df, kernel="radial", 
                 ranges=list(cost=c(0.1, 1, 10, 100, 1000),gamma=c(0.5, 1, 2, 3, 4)))

tune.out$best.model

# Generate ROC plots by increasing variance and increasing bias on the training data

rocplot <- function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

# SVM model for best model with Cost = 10 and Gamma 0.5
svmfit.opt <- svm(y~., data = hearttrain.df, 
                  kernel="radial",
                  gamma=0.5, cost=1,
                  decision.values=T)

fitted <- attributes(predict(svmfit.opt,
                             hearttrain.df,
                             decision.values=TRUE)
)$decision.values

par(mfrow=c(1, 2))
rocplot(fitted, hearttrain.df$y, main="Training Data")

# SVM model and ROC plot for model with increased flexibility by increasing cost
svmfit.flex <- svm(y~., data=hearttrain.df, 
                   kernel="radial",
                   gamma=0.5, 
                   cost=100, 
                   decision.values=T)
fitted=attributes(predict(svmfit.flex,
                          hearttrain.df,
                          decision.values=T)
)$decision.values
rocplot(fitted,hearttrain.df$y, add=T, col="red")

# SVM model and ROC plot for model with increased bias by increasing Gamma
svmfit.bias <- svm(y~., data=hearttrain.df, 
                   kernel="radial",
                   gamma=1, 
                   cost=1, 
                   decision.values=T)

fitted=attributes(predict(svmfit.bias,
                          hearttrain.df,
                          decision.values=T)
)$decision.values
rocplot(fitted,hearttrain.df$y, add=T, col="blue")

# Let's see the power of the model for all the above 3 models on test set
mytable <- table(actual = hearttest.df$y, 
                 predict = predict(tune.out$best.model, hearttest.df))
print(paste('The power of the model with Cost 1 and Gamma 0.5 is', mytable['1', '1']/sum(mytable['1', ])))

mytable <- table(actual = hearttest.df$y, 
                 predict(svmfit.flex,
                         hearttest.df,
                         decision.values=F))
print(paste('The power of the model with Cost 100 and Gamma 0.5 is', mytable['1', '1']/sum(mytable['1', ])))

mytable <- table(actual = hearttest.df$y, 
                 predict(svmfit.bias,
                         hearttest.df,
                         decision.values=F))
print(paste('The power of the model with Cost 1 and Gamma 1 is', mytable['1', '1']/sum(mytable['1', ])))

###########################################Testing the models on test data#########################################

fitted <- attributes(predict(svmfit.opt,
                             hearttest.df,
                             decision.values=T)
)$decision.values
rocplot(fitted,hearttest.df$y, main="Test Data")



fitted <- attributes(predict(svmfit.flex,
                             hearttest.df,
                             decision.values=T)
)$decision.values
rocplot(fitted,hearttest.df$y, add=T, col="red")

fitted <- attributes(predict(svmfit.bias,
                             hearttest.df,
                             decision.values=T)
)$decision.values
rocplot(fitted,hearttest.df$y, add=T, col="blue")

###################Conclusion: Plot with Cost = 1, Gamma = 0.5 shows the best results on the test set##############
