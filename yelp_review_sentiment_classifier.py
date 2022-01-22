# Import libraries and classes required for this example:
import string

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors, datasets

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import csv

# Convert dataset to a pandas dataframe:
Training_Data = pd.read_csv('train_yelp_data.csv')
Test_Data = pd.read_csv('test_yelp_data.csv')


# Data Pre-processing Stage
stopwords = set(stopwords.words('english'))

'''
@text_processing
This function removes all punctuation, stopwords, and performs the Porter Stemmer algorithm
The cleaned text is returned as a list of words.
'''
def text_processing(text):
    remove_punctuation = [
        char for char in text if char not in string.punctuation
    ]
    remove_punctuation = ''.join(remove_punctuation)
    return [
        PorterStemmer().stem(word) for word in remove_punctuation.split()
        if word.lower() not in stopwords
    ]


X = Training_Data['Text']
Y = Training_Data['Class']

#X_test = Test_Data['Text']

# choose the top 10000 words that occur most frequently to be in the vocabulary
# Also transform the yelp review text into a vector format
vector_transform = CountVectorizer(max_features=10000, analyzer=text_processing).fit(X)

X = vector_transform.transform(X)

'''
    #Other machine learning models used to determine which model is most suitable for this problem.
    #Classification Accuracy was compared between the models, with the highest classification accuracy
    #was the determining factor for which model was to be used. 
classifier = neighbors.KNeighborsClassifier(n_neighbors=100)
classifier = RandomForestClassifier(max_depth=100, random_state=0)
'''

# Use Multi-Layer Perceptron (Neural Networks) classifier to fit data:
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

classifier.fit(X_train, Y_train)

predict_Class_Label = classifier.predict(X_test)

print("Multilayer Perceptrion (Neural Network) Classifier Classification Report:")
print(classification_report(Y_test, predict_Class_Label))
