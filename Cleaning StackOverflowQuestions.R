library(tm)
library(SnowballC)
library(wordcloud)
library(MASS)
library(caTools)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(tm.plugin.webmining)

#############Section a ############
## read the file 
Questions = read.csv("ggplot2questions2016_17.csv", stringsAsFactors=FALSE)
str(Questions)

# We want to predict if a question is useful
# Lets create a new variable called "Useful" that converts the 
# score number to useful (or not)
# anything greater than or equal to one is useful
Questions$Useful = as.factor(as.numeric(Questions$Score >= 1))
# And we remove the old "Score" column - we won't use it anymore
Questions$Score <- NULL
str(Questions)

table(Questions$Useful) #Regarding the table, our baseline model would be to predict that a question is always useful

#Clean Body : Remove html syntax from body text
Questions$body_clean = body_clean=lapply(Questions$Body, extractHTMLStrip)
#Delete Body independent variable because now we use body_clean
Questions$Body <- NULL


# Step 1: Convert Body_clean and title to a "corpus"
# A vector source interprets each element of the vector as a document.
# Corpus creates a collection of documents 
corpusBody = Corpus(VectorSource(Questions$body_clean))
# The body are now "documents"
corpusBody[[1]]
strwrap(corpusBody[[1]])
corpusTitle = Corpus(VectorSource(Questions$Title))
corpusTitle[[1]]


# Step 2: Change all the text to lower case.
corpusBody = tm_map(corpusBody, tolower)
strwrap(corpusBody[[1]])
corpusTitle = tm_map(corpusTitle, tolower)

# Step 3: Remove all punctuation
corpusBody = tm_map(corpusBody, removePunctuation)
strwrap(corpusBody[[1]])
corpusTitle = tm_map(corpusTitle, removePunctuation)


# Step 4: Remove stop words
# Remove stopwords and "ggplot2" - this is a word common to all of our bodies and titles
corpusBody = tm_map(corpusBody, removeWords, c("ggplot", stopwords("english")))
strwrap(corpusBody[[1]])
corpusTitle = tm_map(corpusTitle, removeWords, c("ggplot", stopwords("english")))

# Step 5: Stem our document
corpusBody = tm_map(corpusBody, stemDocument)
strwrap(corpusBody[[1]])
corpusTitle = tm_map(corpusTitle, stemDocument)

#we remove numbers from Body
corpusBody <- tm_map(corpusBody, removeNumbers)

# Step 6: Create a word count matrix (rows are titles/bodies, columns are words)
# We've finished our basic cleaning, so now we want to calculate frequencies
# of words across the bodies and titles
frequenciesBody = DocumentTermMatrix(corpusBody)
frequenciesBody
frequenciesTitle= DocumentTermMatrix(corpusTitle)

# Step 7: Account for sparsity
# We currently have way too many words, which will make it hard to train
# our models and may even lead to overfitting.
# Use findFreqTerms to get a feeling for which words appear the most

# Words that appear at least 50 times in Body:
findFreqTerms(frequenciesBody, lowfreq=500) #193 words
# Words that appear at least 20 times in Body:
findFreqTerms(frequenciesBody, lowfreq=20) #1000 words

# Words that appear at least 50 times in Title:
findFreqTerms(frequenciesTitle, lowfreq=50) #134 words
# Words that appear at least 20 times in Title :
findFreqTerms(frequenciesTitle, lowfreq=20) #319 words

# Our solution to the possibility of overfitting is to only keep terms
# that appear in 3,5% or more of the bodies and 1% or more of the titles. 

sparseBody = removeSparseTerms(frequenciesBody, 0.93)
sparseBody
sparseTitle = removeSparseTerms(frequenciesTitle, 0.95)

# Step 8: Create data frame from the document-term matrix
TitleTM = as.data.frame(as.matrix(sparseTitle))
# We have some variable names that start with a number, 
# which can cause R some problems. Let's fix this before going
# any further
colnames(TitleTM) = make.names(colnames(TitleTM))

BodyTM = as.data.frame(as.matrix(sparseBody))
colnames(BodyTM) = make.names(colnames(BodyTM))

# This isn't our original dataframe, so we need to bring that column
# with the dependent variable into this new one
TitleTM$Useful = Questions$Useful


#Combine the corpus-body and corpus_title such that it specifies if
#a word is from the title or body during the model building phase.
#use cbind and attach a prefix to each word that is a column name in the data frame matrix.

colnames(BodyTM) <- paste("b", colnames(BodyTM), sep = "_")
colnames(TitleTM) <- paste("t", colnames(TitleTM), sep = "_")

FinalTM <- cbind(BodyTM, TitleTM)
FinalTM$Useful = Questions$Useful
FinalTM$t_Useful <- NULL
FinalTM$b_ggplot <- NULL
FinalTM$t_ggplot2 <- NULL


########### Section b ############
## We finished ceaning the dataset, so now we will train the models

set.seed(123) 
spl = sample.split(FinalTM$Useful, SplitRatio = 0.7)

QuestionsTrain = FinalTM %>% filter(spl == TRUE)
QuestionsTest = FinalTM %>% filter(spl == FALSE)

#Baseline Model accuracy : 0,51
table(Questions$Useful)
3791/(3791+3677)

###Logistic regression model

logregmod <- glm(Useful~ ., data=QuestionsTrain, family="binomial")
summary(logregmod)

# Predictions on test set
PredictLog = predict(logregmod, newdata = QuestionsTest, type = "response")
table(QuestionsTest$Useful, PredictLog > 0.5)

####Cross-validated CART model

train.cart <- train(Useful ~.,
                    data = QuestionsTrain,
                    method = "rpart",
                    tuneGrid = data.frame(cp = seq(0, .04, by=.002)),
                    trControl = trainControl(method = "cv", number=10),
                    metric = "Accuracy")

# look at the cross validation results, stored as a data-frame
train.cart$results 
train.cart

# plot the results
ggplot(train.cart$results, aes(x=cp, y=Accuracy)) + geom_point(size=3) +
  xlab("Complexity Parameter (cp)") + geom_line()

# Extract the best model and make predictions
train.cart$bestTune
mod.cart = train.cart$finalModel
prp(mod.cart, digits=3)

#we predict on the test set
pred.cart = predict(mod.cart, newdata=QuestionsTest, type="class")
table(QuestionsTest$Useful, pred.cart)


### Linear Discriminant Analysis

library(MASS)
lda.mod = lda(Useful ~ ., data = QuestionsTrain)

predict.lda = predict(lda.mod, newdata = QuestionsTest)$class
table(QuestionsTest$Useful, predict.lda)

#### Basic Random Forests model 
RFmod = randomForest(Useful ~ ., data=QuestionsTrain)

PredictRF = predict(RFmod, newdata = QuestionsTest)
table(QuestionsTest$Useful, PredictRF)
#We obtain not a really good accuracy so we use cross validation to select the parameter mtry. 

RFmodCV = train(Useful ~ .,                 data = QuestionsTrain,
                 method = "rf",
                 tuneGrid = data.frame(mtry = 1:120),
                 trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))
RFmodCV
RFmodCV$results

ggplot(RFmodCV$results, aes(x = mtry, y = Accuracy)) + geom_point(size = 2) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

mod.rf = RFmodCV$finalModel
predict.rf = predict(mod.rf, newdata = QuestionsTest)
table(QuestionsTest$Useful, predict.rf)

ranger_df_lda = data.frame(labels = QuestionsTest$Useful, predictions = predict.rf, baseline = 0,5076)
boot_all_metrics(ranger_df_lda, 1:2240)

###Bootstrap to assess the performance of the LDA model

library(boot)

tableAccuracy <- function(label, pred) {
  t = table(label, pred)
  a = sum(diag(t))/length(label)
  return(a)
}

tableTPR <- function(label, pred) {
  t = table(label, pred)
  return(t[2,2]/(t[2,1] + t[2,2]))
}

tableFPR <- function(label, pred) {
  t = table(label, pred)
  return(t[1,2]/(t[1,1] + t[1,2]))
}

boot_accuracy <- function(data, index) {
  labels <- data$label[index]
  predictions <- data$prediction[index]
  return(tableAccuracy(labels, predictions))
}

boot_tpr <- function(data, index) {
  labels <- data$label[index]
  predictions <- data$prediction[index]
  return(tableTPR(labels, predictions))
}

boot_fpr <- function(data, index) {
  labels <- data$label[index]
  predictions <- data$prediction[index]
  return(tableFPR(labels, predictions))
}

boot_all_metrics <- function(data, index) {
  acc = boot_accuracy(data, index)
  tpr = boot_tpr(data, index)
  fpr = boot_fpr(data, index)
  return(c(acc, tpr, fpr))
}

# sanity test
table(QuestionsTest$Useful, predict.lda)
ranger_df_lda = data.frame(labels = QuestionsTest$Useful, predictions = predict.lda, baseline = 0,5076)
boot_all_metrics(ranger_df_lda, 1:2240)

ranger_boot_lda = boot(ranger_df_lda, boot_all_metrics, R = 100000)
ranger_boot_lda
boot.ci(ranger_boot_lda, index = 1, type = "basic")
boot.ci(ranger_boot_lda, index = 2, type = "basic")
boot.ci(ranger_boot_lda, index = 3, type = "basic")



