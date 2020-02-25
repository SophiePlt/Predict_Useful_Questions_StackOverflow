# Predict_Useful_Questions_StackOverflow
Get a sense of whether a question posted on StackOverflow is useful or not

The “score” of a question is equal to the total number of upvotes for that question minus the total number of downvotes.
In this problem, I work with question data from Stack Overflow, made available as part of the Stack exchange Network “data dump”. 
The particular dataset that I work with contains questions about the ggplot2 package in the R programming language, which were posted 
between January 2016 and September 2017. There are 7,468 observations, with each observation recording text data associated with both 
the title and the body of the associated question and also the score of the question.

After a question is posted, it would be beneficial for Stack Overflow to get an immediate sense of whether the question is useful or not.
With such knowledge, the website could automatically promote questions that are believed to be useful to the top of the page.
With this goal in mind, in this problem I build models to predict whether or not a ggplot2 question is useful, 
based on the title and body text data associated with each question. 
To be concrete, I say that a question is useful if its score is greater than or equal to one.

I start by cleaning the dataset and then I train different models: Logistic regression models, CART model, LDA model 
and Random Forest Model. 
