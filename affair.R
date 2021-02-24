# Load the Dataset

claimants <- read.csv(file.choose()) # Choose the claimants Data set
attach(claimants)
sum(is.na(claimants))
claimants=claimants[,11:17]
claimants1=claimants
# NA values are present in CLMSEX, CLMINSUR, SEATBELT, CLMAGE
###########

colnames(claimants)
# Removing the first column which is is an Index

# Preparing a linear regression 
mod_lm <- lm(yrsmarr6~ ., data = claimants)
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
model <- glm(yrsmarr6 ~ ., data = claimants, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))


# Build Model on 100% of data
# Removing the first column which is is an Index
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(yrsmarr6 ~ ., data = claimants, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, claimants, type = "response")
prob_full
# Check multicollinearity in the model
library(car)
vif(fullmodel)

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(claimants$yrsmarr6, prob_full, threshold = optCutOff)

# Confusion Matrix
###################

# Data Partitioning
claimants_test=claimants
claimants_train=claimants
n <-  nrow(claimants1)
n1 <-  n * 0.78
n2 <-  n - n1
train_index <- sample(1:n, n1)
train <- claimants1[train_index, ]
test <-  claimants1[-train_index, ]
train$slghtrel=NULL
test$slghtrel=NULL
# Train the model using Training data

mod_lm 
summary(mod_lm)
# Prediction on test data
prob_test <- predict( mod_lm , newdata = claimants_train, type = "response")
prob_test


# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(mod_lm, newdata = claimants_test, type = "response")
prob_train

