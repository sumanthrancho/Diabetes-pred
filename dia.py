

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection and Analysis

PIMA Diabetes Dataset
"""

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# printing the first 5 rows of the dataset
diabetes_dataset.head()

# number of rows and Columns in this dataset
diabetes_dataset.shape

# getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

"""0 --> Non-Diabetic

1 --> Diabetic
"""

diabetes_dataset.groupby('Outcome').mean()

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

"""Train Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Training the Model"""

classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

"""Model Evaluation

Accuracy Score
"""

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""Making a Predictive System"""

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

"""Saving the trained model"""

import pickle

filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))



# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')



# import numpy as np
# import pickle
# import streamlit as st


# # loading the saved model
# loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# # creating a function for Prediction

# def diabetes_prediction(input_data):


#     # changing the input_data to numpy array
#     input_data_as_numpy_array = np.asarray(input_data)

#     # reshape the array as we are predicting for one instance
#     input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#     prediction = loaded_model.predict(input_data_reshaped)
#     print(prediction)

#     if (prediction[0] == 0):
#       return 'The person is not diabetic'
#     else:
#       return 'The person is diabetic'



# def main():


#     # giving a title
#     st.title('Diabetes Prediction Web App')


#     # getting the input data from the user


#     Pregnancies = st.text_input('Number of Pregnancies')
#     Glucose = st.text_input('Glucose Level')
#     BloodPressure = st.text_input('Blood Pressure value')
#     SkinThickness = st.text_input('Skin Thickness value')
#     Insulin = st.text_input('Insulin Level')
#     BMI = st.text_input('BMI value')
#     DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
#     Age = st.text_input('Age of the Person')


#     # code for Prediction
#     diagnosis = ''

#     # creating a button for Prediction

#     if st.button('Diabetes Test Result'):
#         diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])


#     st.success(diagnosis)


# if __name__ == '__main__':
#     main()
