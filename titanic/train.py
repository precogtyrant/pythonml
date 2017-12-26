#Predicting survivors on the titanic dataset using Logistic Regression module of SciKit Learn
#Prediction accuracy score consistently between 80% to 84%, by randomly splitting the 
#data into train/test sets using sklearn's train_test_split module on a 80/20 basis
#accuracy is 85% when taking the last 20% of the data as the testing data and the remaining 
#as the training data

# %matplotlib inline
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
def get_title(name):
    """modify titles"""
    title = name.split(", ")[1].split()[0]
    title.replace("Mlle.","Miss.")
    valid_titles = ["Mr.","Mrs.","Miss.","Master."]
    if(title not in valid_titles):
        title = "Mr."
    title



titanic_data = pd.read_csv("titanic.csv")


titanic_data["Title"] = titanic_data["Name"].apply(lambda x: get_title(x)).fillna("Mr.")
titanic_data["Title"] = pd.get_dummies(titanic_data["Title"])
titanic_data["Sex"] = pd.get_dummies(titanic_data["Sex"])
titanic_data["Age"].fillna(titanic_data["Age"].mean(), inplace = True)
titanic_data["Pclass"] = pd.get_dummies(titanic_data["Pclass"])

#using the first letter of the cabin as the cabin identifier and setting the one with the highest frequency
#as the cabin class for those rows where its missing(it was missing in over 600 rows)
titani_cabin_mode = titanic_data["Cabin"].dropna().apply(lambda x: x[0]).mode() #cabin with highest frequency
titanic_data["Cabin"] = titanic_data["Cabin"].dropna().apply(lambda x: x[0]) #dropna so that function can be applied without error
titanic_data["Cabin"].fillna(titani_cabin_mode[0],inplace=True)
titanic_data["Cabin"] = pd.get_dummies(titanic_data["Cabin"])

titanic_embarked_mode = titanic_data["Embarked"].dropna().mode()
titanic_data["Embarked"].fillna(titanic_embarked_mode,inplace=True)
titanic_data["Embarked"] = pd.get_dummies(titanic_data["Embarked"])

#train,test = titanic_data[:-20],titanic_data[-20:] #80/20 split manually
train,test = train_test_split(titanic_data,test_size = 0.2)

model = LogisticRegression();
train_x = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Title','Embarked','Cabin']]
test_x = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Title','Embarked','Cabin']]
train_Y = train["Survived"]
model.fit(train_x,train_Y)
prediction = model.predict(test_x)
prediction_result = pd.DataFrame({"Survived":prediction,"PassengerId":test["PassengerId"], "Name":test["Name"]})
prediction_result["Survived"] = prediction_result["Survived"]
print(accuracy_score(test["Survived"],prediction.astype(int)))