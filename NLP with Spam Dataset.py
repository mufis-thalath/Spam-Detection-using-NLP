# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 08:43:08 2021

@author: mufis
"""

"""
import os
os.chdir(r'C:\Users\dines\Desktop\Machine learning project')
os.getcwd()
"""

##########-----------------NLP with Spam Dataset---------------------#########

### Importing and reading the Dataset
import pandas as pd
Dataset = pd.read_csv('SMSSpamCollection', sep='\t', names=['label','messages'])


### Cleaning the dataset and preprocessing
import re                  # This is regular expression library
import nltk                # This is the main library for NLP
nltk.download('stopwords')
from nltk.corpus import stopwords            # This is for removing all unwanted stopwords (Eg:in, a, etc)
from nltk.stem.porter import PorterStemmer   # This is for stemming the words by using its common sound
from nltk.stem import WordNetLemmatizer      # This is same as stemming but lemmatizer will give us an actual meaningful root word
PS = PorterStemmer()                         # Creating an instance of PorterStemmer
Lem = WordNetLemmatizer()                    # Creating an instance of Lemmatizer

#A-Cleaning & Preprocessing (Using PorterStemmer)
Corpus = []                                                    # Putting the cleaned and preprocessed dataset inside corpus 
for i in range(0, len(Dataset)):                               # specifying the entire length of the dataset from 0 record to 5572 record
    review = re.sub('[^a-zA-Z]', ' ', Dataset['messages'][i])  # here ^ means except any charcter between a-z and A-Z replace it with ' '(space) and do it in the messages column of the dataset with 'i' representing each record
    review = review.lower()                                    # this will get all the words to lowercase to avoid duplicates
    review = review.split()                                    # this will split all the words and will give us a list of words to be put in review
    review = [PS.stem(word) for word in review if not word in stopwords.words('english')]    # here we do stemming on the words present in review but omitting those words that are stopwords.words(english), so this will result in getting the base form of each non-stopword
    review = ' '.join(review)                                  # this will join all the stemmed words (from the previous review) into sentences with ' '(spaces)
    Corpus.append(review)                                      # this will put review into the previously made list named Corpus

#B-Cleaning & Preprocessing (Using WordNetLemmatizer)
Corpus = []                                                    # Putting the cleaned and preprocessed dataset inside corpus 
for i in range(0, len(Dataset)):                               # specifying the entire length of the dataset from 0 record to 5572 record
    review = re.sub('[^a-zA-Z]', ' ', Dataset['messages'][i])  # here ^ means except any charcter between a-z and A-Z replace it with ' '(space) and do it in the messages column of the dataset with 'i' representing each record
    review = review.lower()                                    # this will get all the words to lowercase to avoid duplicates
    review = review.split()                                    # this will split all the words and will give us a list of words to be put in review
    review = [Lem.lemmatize(word) for word in review if not word in stopwords.words('english')]    # here we do stemming on the words present in review but omitting those words that are stopwords.words(english), so this will result in getting the base form of each non-stopword
    review = ' '.join(review)                                  # this will join all the stemmed words (from the previous review) into sentences with ' '(spaces)
    Corpus.append(review)                                      # this will put review into the previously made list named Corpus


### Creating Bag of Words model using CountVectorizer from SKLearn library
""" Bag of Words will create a document matrix with respect to the words inside Corpus. Since we cannot
    just give the words for creating the model, we need to convert it to a numerical format (called as 
    Vectors in NLP) for training the model. This is what Bag of Words model does (For eg: in sentiment analysis).
"""
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer(max_features=5000)                        # Initializing and making an instance of CountVeactorizer
''' Here we take only 5000 features instead of all 7098 features because some features (or words in this case)
    may only be appearing once or twice and will not be more frequently occuring. So we can eliminate such 
    less frequently occuring words and keep only the most frequently occuring words by adding max_features
    and selecting the number of features we need. This can be any number and we have chosen 5000 features in this case)
'''
# Fitting the Bag of words model into Corpus and creating the X (independent variable) with only the vectorized feature
X = CV.fit_transform(Corpus).toarray()                         # Transforming the Corpus with CountVEctorizer and converting into an Array object
# After getting our X variable from the Corpus, now we need to get the dependent Y variable from the Dataset which will basically be the Label feature
Y = pd.get_dummies(Dataset['label'], drop_first=True)          # Since the label column was categorical (having ham and spam) we will use get_dummies encoding to convert it to binary representation of 0 and 1
# Since Y is a dataframe, we need to convert it to an array. We will do that using iloc function and just getting the values out of it
Y = Y.iloc[:,:].values


### Splitting into train and test test for modelling
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)


### Training the model using Naive Bayes Classifier (Since naive bayes is a classification technique that worls completely on probability, it is a good classifier for NLP)
from sklearn.naive_bayes import MultinomialNB                  # We choose multinomial because it works for any number of classes
MultiNB = MultinomialNB()
spam_detection = MultiNB.fit(X_train, Y_train)                 # Fitting the NB model on the train set (x_train and y_train)


### Predicting on the test set
Y_pred = spam_detection.predict(X_test)                        # Using the trained spam_detection model to predict values in the test set (x_test)


### Now since we have the y_pred, we need to compare the values of y_pred with the values of y_test to see how accurate our model has performed in the test set
# For this, we will see the confusion matrix and accuracy score from sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
conf_matrix = confusion_matrix(Y_pred, Y_test)
accuracy = accuracy_score(Y_pred, Y_test)                      # 0.9842067480258435 (98% accuracy score which is very good)



