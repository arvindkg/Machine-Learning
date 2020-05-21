# ------------------------------------------------ 
# Classification of Forest Cover Type Using Random Forests Algorithm
#
# Data Source: https://archive.ics.uci.edu/ml/datasets/covertype
#
# Author: Arvind Kumar
# Email: arvindk.cse@gmail.com
#-------------------------------------------------


import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the training and test data sets
data = pd.read_csv('./UCI_covtype.data', header=None)

# Analyze the data
#data.describe()
print(data.shape)

# Create numpy arrays for use with scikit-learn
train_X = data.iloc[:, :-1]
train_y = data.iloc[:, 54:55]

# Split the training set into training and validation sets
X_train,X_test,y_train,y_test = train_test_split(train_X,train_y,test_size=0.2)

# Train and predict with the random forest classifier
rf_classifier = ensemble.RandomForestClassifier()

# train rf_classifier
rf_classifier.fit(X_train,y_train.values.ravel())

# predict using trained rf_classifier
y_rf = rf_classifier.predict(X_test)

print("============RandomForestClassifier Accuracy: ")

print( metrics.accuracy_score(y_test,y_rf) )

