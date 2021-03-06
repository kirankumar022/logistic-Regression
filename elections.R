# Load the Dataset
election <- read.csv(file.choose()) # Choose the claimants Data set

sum(is.na(election))

# Omitting NA values from the Data 
elections1 <- na.omit(election ) # na.omit => will omit the rows which has atleast 1 NA value
dim(election )

# Alternatively We can apply mean/median/mode imputation
elections1 <- election  # work with original data for imputation
summary(elections1)

# NA values are present in CLMSEX, CLMINSUR, SEATBELT, CLMAGE
###########

colnames(election)
election  <- election [ , -1] # Removing the first column which is is an Index
colnames(election)
attach(election)
election$Year=NULL


# Preparing a linear regression 
mod_lm <- lm(Result~ ., data =election )
summary(mod_lm)

pred1 <- predict(mod_lm, election)
pred1
# plot(claimants$CLMINSUR, pred1)

# We can also include NA values but where ever it finds NA value
# probability values obtained using the glm will also be NA 
# So they can be either filled using imputation technique or
# exlclude those values 


# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(Result ~ ., data = election, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Predicition to check model validation
prob <- predict(model, election, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, election))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, election$Result)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)

library(caret)
library(e1071)

# Confusion Matrix
confusionMatrix(factor(election$Result, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))


# Build Model on 100% of data
elections1 <- election[ , -1] # Removing the first column which is is an Index
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(Result ~ ., data = elections1, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, elections1, type = "response")
prob_full


# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(elections1$Result, prob_full)
optCutOff

# Check multicollinearity in the model
library(car)
vif(fullmodel)

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(elections1$Result, prob_full, threshold = optCutOff)


# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(election$Result, prob_full)


# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

expected <- factor(elections1$Result)
predicted <- factor(predvalues)
sensitivity()
?confusionMatrix()
###################
# Data Partitioning
n <-  nrow(elections1)
n1 <-  n * 0.85
n2 <-  n - n1
train_index <- sample(1:n, n1)
train <- election[train_index, ]
test <-  election[-train_index, ]
train$Amount.Spent=NULL
test$Amount.Spent=NULL
# Train the model using Training data
finalmodel <- glm(Result ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test


# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$Result, test$pred_values)


# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train



