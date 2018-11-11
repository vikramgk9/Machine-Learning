rm(list=ls())
student.por <- read.table("student-por.csv", sep=";", header = T, stringsAsFactors = T)
#### Preprocess Student Portuguese Data (student.por) ####
student.por$famrel <- factor(student.por$famrel, order = T)
student.por$traveltime <- factor(student.por$traveltime, order = T)
student.por$studytime <- factor(student.por$studytime, order = T)
student.por$freetime <- factor(student.por$freetime, order = T)
student.por$failures <- factor(student.por$failures, ordered = T)
student.por$goout <- factor(student.por$goout, order = T)
student.por$Walc <- factor(student.por$Walc, order = T)
student.por$Dalc <- factor(student.por$Dalc, order = T)
student.por$health <- factor(student.por$health, order = T)
student.por$Medu <- factor(student.por$Medu, order = T)
goout0 <- data.frame(model.matrix(~goout-1, data = student.por))
schoolsup0 <- data.frame(model.matrix(~schoolsup-1, data = student.por))
schoolstudytime0 <- data.frame(model.matrix(~studytime-1, data = student.por))
schoolfreetime0 <- data.frame(model.matrix(~freetime-1, data = student.por))
student.por <- cbind(student.por, goout0, schoolsup0, schoolstudytime0, schoolfreetime0)
student.por[which(names(student.por) == "goout")] <- NULL
student.por[which(names(student.por) == "schoolsup")] <- NULL
student.por[which(names(student.por) == "studytime")] <- NULL
student.por[which(names(student.por) == "freetime")] <- NULL
Mode <- function(x, na.rm = FALSE) {if(na.rm){x = x[!is.na(x)]}
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}
for (var in 1:ncol(student.por)) {
  if (class(student.por[,var])=="numeric") {
    student.por[is.na(student.por[,var]),var] <- mean(student.por[,var], na.rm = TRUE)
  } else if (class(student.por[,var]) %in% c("character", "factor")) {
    student.por[is.na(student.por[,var]),var] <- Mode(student.por[,var], na.rm = TRUE)}
}
#### Linear Regression for student.por ####
set.seed(6702)
n <- nrow(student.por)
train.indices <- sample(1:n, .7 * nrow(student.por))
test.indices <- setdiff(1:n, train.indices)
train.por <- student.por[train.indices, ]
test.por <- student.por[test.indices, ]
lm.por <- lm(G3 ~ ., data = train.por)
summary(lm.por)
lm.sig.por <- lm(G3 ~ G2 + G1 + failures + Dalc + health + goout1 + goout4 + reason,
                 data = train.por)
summary(lm.sig.por)
lm.predictions.porsig <- predict(lm.sig.por, newdata = test.por)
MSE.por <- mean((test.por$G3 - lm.predictions.porsig) ^ 2)
MSE.por
interval.high <- mean(student.por$G3) + 2 * sqrt(MSE.por)
interval.low <- mean(student.por$G3) - 2 * sqrt(MSE.por)
interval.high
interval.low
#### Decision Tree Classification for student.por ####
rm(list=ls())
require(rpart)
require(rattle)
student.por <- read.csv('student-por.csv', sep = ";", header = T)
student.por$result <- rep('Pass', nrow(student.por))
student.por$result[which(student.por$G3 < 10)] <- 'Fail'
student.por$result <- as.factor(student.por$result)
student.por$G3 <- NULL
set.seed(6702)
n <- nrow(student.por)
train.indices <- sample(1:n, .7 * nrow(student.por))
test.indices <- setdiff(1:n, train.indices)
train.df <- student.por[train.indices, ]
test.df <- student.por[test.indices, ]
mymodel.max <- rpart(result ~ ., data = train.df, method="class",
                     parms=list(split="information"), 
                     control=rpart.control(usesurrogate=0,
                                           maxsurrogate=0, 
                                           cp=0, 
                                           minbucket=1, 
                                           minsplit=2))
fancyRpartPlot(mymodel.max, main="Maximal Decision Tree")
print(mymodel.max$cptable)
plotcp(mymodel.max)
xerr <- mymodel.max$cptable[, "xerror"]
minxerr <- which.min(xerr)
minxerr
mincp <- mymodel.max$cptable[minxerr, "CP"]
mincp
mymodel.max.prune <- prune(mymodel.max, cp=mincp)  
fancyRpartPlot(mymodel.max.prune, main="Decision Tree With Minimum C.V. Error")
asRules(mymodel.max.prune)
mymodel.max.prune.predict <- predict(mymodel.max.prune, 
                                     newdata=test.df, 
                                     type="class")
mytable <- table(test.df$result, mymodel.max.prune.predict, 
                 dnn=c("Actual", "Predicted"))
mytable
round(100*mytable/length(mymodel.max.prune.predict))
(mytable[1,1] + mytable[2,2])/sum(mytable)
#### Random Forest Classification for student.por ####
require('randomForest')
require('pROC')
require('verification')
set.seed(6702)
rf <- randomForest(result ~ .,data = student.por[train.indices,],
                   ntree=500,
                   mtry=4,
                   importance=TRUE, 
                   na.action=na.roughfix,
                   replace=T)
rf
rn <- round(importance(rf), 2)
rn[order(rn[, 3], decreasing=TRUE), ]
varImpPlot(rf, main="Variable Importance in the Random Forest")
roc(rf$y, as.numeric(rf$predicted))$auc
ci.auc(rf$y, as.numeric(rf$predicted))
roc.plot(as.integer(as.factor(student.por[train.indices, "result"])) - 1,
         rf$votes[, 2], 
         xlab = "Type I Error Rate", 
         ylab = "Power",
         main="OOB ROC Curve for the Random Forest")
set.seed(6702)
rf <- randomForest(result ~ .,data=student.por[train.indices,], 
                   ntree=500,
                   mtry=4, 
                   importance=TRUE, na.action=na.roughfix, 
                   replace=FALSE, 
                   sampsize=c(35,35),
                   cutoff=c(0.4,0.2))
rf
roc(rf$y, as.numeric(rf$predicted))$auc
pred.test <- predict(rf, newdata=student.por[test.indices,])
mytable <- table(student.por[test.indices,]$result, pred.test,dnn=c("Actual", "Predicted"))
round(100 * mytable / sum(mytable))