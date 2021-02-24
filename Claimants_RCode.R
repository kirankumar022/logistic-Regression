# Load the Dataset
claimants <- read.csv(file.choose()) # Choose the claimants Data set

sum(is.na(claimants))

# Omitting NA values from the Data 
claimants1 <- na.omit(claimants) # na.omit => will omit the rows which has atleast 1 NA value
dim(claimants1)

# Alternatively We can apply mean/median/mode imputation
claimants1 <- claimants # work with original data for imputation
summary(claimants1)

# NA values are present in CLMSEX, CLMINSUR, SEATBELT, CLMAGE

# Mean imputation for continuous data - CLMAGE
claimants1$CLMAGE[is.na(claimants1$CLMAGE)] <- mean(claimants1$CLMAGE, na.rm = TRUE)


# Mode imputation for categorical data
# Custom function to calculate Mode
Mode <- function(x){
     a = table(x) # x is a vector
     names(a[which.max(a)])
}

claimants1$CLMSEX[is.na(claimants1$CLMSEX)] <- Mode(claimants1$CLMSEX[!is.na(claimants1$CLMSEX)])
claimants1$CLMINSUR[is.na(claimants1$CLMINSUR)] <- Mode(claimants1$CLMINSUR[!is.na(claimants1$CLMINSUR)])
claimants1$SEATBELT[is.na(claimants1$SEATBELT)] <- Mode(claimants1$SEATBELT[!is.na(claimants1$SEATBELT)])

# We can also use imputeMissings package for imputation

sum(is.na(claimants1))
dim(claimants1)
###########

colnames(claimants)
claimants <- claimants[ , -1] # Removing the first column which is is an Index

# Preparing a linear regression 
mod_lm <- lm(ATTORNEY ~ ., data = claimants)
summary(mod_lm)

pred1 <- predict(mod_lm, claimants)
pred1
# plot(claimants$CLMINSUR, pred1)

# We can also include NA values but where ever it finds NA value
# probability values obtained using the glm will also be NA 
# So they can be either filled using imputation technique or
# exlclude those values 


# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(ATTORNEY ~ ., data = claimants, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Predicition to check model validation
prob <- predict(model, claimants, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, claimants))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, claimants$ATTORNEY)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix
confusionMatrix(factor(claimants$ATTORNEY, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))


# Build Model on 100% of data
claimants1 <- claimants1[ , -1] # Removing the first column which is is an Index
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(ATTORNEY ~ ., data = claimants1, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, claimants1, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(claimants1$ATTORNEY, prob_full)
optCutOff

# Check multicollinearity in the model
library(car)
vif(fullmodel)

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(claimants1$ATTORNEY, prob_full, threshold = optCutOff)


# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(claimants1$ATTORNEY, prob_full)


# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

confusionMatrix(factor(predvalues, levels = c(0, 1)), factor(claimants1$ATTORNEY, levels = c(0, 1)))

expected <- factor(claimants1$ATTORNEY)
predicted <- factor(predvalues)
results <- confusionMatrix(actuals = expected, predictedScores = predicted)

sensitivity()
confusionMatrix(actuals = claimants1$ATTORNEY, predictedScores = predvalues)
?confusionMatrix()

###################
# Data Partitioning
n <-  nrow(claimants1)
n1 <-  n * 0.85
n2 <-  n - n1
train_index <- sample(1:n, n1)
train <- claimants1[train_index, ]
test <-  claimants1[-train_index, ]

# Train the model using Training data
finalmodel <- glm(ATTORNEY ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$ATTORNEY)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$ATTORNEY, test$pred_values)


# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$ATTORNEY)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train

