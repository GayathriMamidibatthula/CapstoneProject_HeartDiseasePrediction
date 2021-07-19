#1. LOAD LIBRARIES AND IMPORT DATASET
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(naniar)) install.packages("naniar", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(readr)
library(ggplot2)
library(naniar)
library(caTools)
library(caret)
library(e1071)
library(randomForest)
#Importing Dataset
setwd("C:/Users/gayathri_sri_mamidib/Desktop/Gayathri Imp/HDSC/Capstone2 - Heart Disease Prediction")
df<- read_csv("heart.csv")
head(df)

#Examining the data
str(df)

#Some of the data is in integer format. Converting int variable to factors

df$sex <- as.factor(df$sex)
df$cp <- as.factor(df$cp)
df$fbs <- as.factor(df$fbs)
df$restecg <- as.factor(df$restecg)
df$exang <- as.factor(df$exang)
df$slope <- as.factor(df$slope)
df$ca <- as.factor(df$ca)
df$thal <- as.factor(df$thal)
df$target <- as.factor(df$target)

str(df)

#Analyze no of people suffering from heart disease
table(df$target)

#2. EXPLORATORY DATA ANALYSIS(EDA)

#Plot heart disease count by Gender
#Variables:
#1. Sex: Male - 1, Female - 0
#2. Target(whether patient has heart disease or not): Yes - 1, No - 0 
counts <- table(df$target, df$sex)

barplot(counts, main = "Heart Disease by Gender category", xlab = "Gender",ylab="Heart Disease Count", col = c("green","red"), beside = TRUE )

#Plot Heart Disease Count by Fasting Blood Sugar

fbs <- table(df$target, df$fbs)

barplot(fbs, main = "Heart Disease by Fasting Blood Sugar Category", xlab = "Fasting Blood Sugar", col = c("green","red"), legend = rownames(fbs), beside = TRUE )

#Heart Disease by Chest Pain category

cp <- table(df$target, df$cp)

barplot(cp, main = "Heart Disease by Chest Pain category ", xlab = "Chest Pain category ",
        col = c("green","red"), legend =  rownames(cp), beside = TRUE )

#Relation between cholestrol and chest pain

ggplot(df, aes(x = cp, y = chol)) + geom_boxplot(aes(fill = target),position = position_dodge(0.9)) + scale_fill_manual(values = c("#999999", "#E69F00")) + ggtitle("Cholestrol Vs Chest Pain")

#3. DATA CLEANING

#Handling Missing Data:

vis_miss(df)

#No missing data is observed.

#4. BUILDING THE ML MODELS

# a) Logistic Regression:

#Splitting the data into training set and test set:
set.seed(123)
split = sample.split(df$target, SplitRatio = 0.8)
training_set = subset(df, split = TRUE)
test_set = subset(df, split = FALSE)

#Split ratio - 80% training data and 20% test data. The 80/20 split is done to prevent overfitting of the model

#Scaling numeric values:

training_set[ , c(1,4,5,8, 10)] =  scale(training_set[, c(1,4,5,8, 10)])
test_set[ , c(1,4,5,8, 10)] = scale(test_set[ , c(1,4,5,8, 10)])

# a) SVM(Support Vector Machine) Model:

#Fitting SVM to training set using linear kernel:

classifier_svm = svm(formula = target ~ .,
                     data = training_set,
                     type = 'C-classification',
                     kernel = 'linear')

# Predicting the Test set results
y_pred_svm = predict(classifier_svm, newdata = test_set[-14])

confusionMatrix(y_pred_svm , test_set$target)

#Test Results:
#Accuracy: 87.79%
#Sensitivity/Recall: 81.16%
#Specificity: 93.33%

# b) Fitting SVM to training set using radial kernel:
classifier_svm = svm(formula = target ~ .,
                     data = training_set,
                     type = 'C-classification',
                     kernel = 'radial')

# Predicting the Test set results
y_pred_svm = predict(classifier_svm, newdata = test_set[-14])

confusionMatrix(y_pred_svm , test_set$target)

#Test Results:
#Accuracy: 86.14%
#Sensitivity/Recall: 84.06%
#Specificity: 87.88%

#Linear kernel is better than radial kernel for SVM. Hence, classification is linear here.

# c) Random Forest

#Fitting Random Forest Model

classifier_rf = randomForest(x = training_set[-14], y = training_set$target, ntree = 400)

# Predicting the Test set results
y_pred_rf = predict(classifier_rf, newdata = test_set[-14])

confusionMatrix(y_pred_rf , test_set$target)
#Test Results:
#Accuracy: 100%
#Sensitivity/Recall: 100%
#Specificity: 100%

#Since accuracy is 100%, checking possibility of overfitting

#Check possibility of overfitting using repeated k-cross validation:

# define training control
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(target~., data=df, trControl=train_control, method="nb")
# summarize results
print(model)

#After performing repeated k-fold cross validation, the final accuracy is around 82.2%, showing that model was overfitted.





