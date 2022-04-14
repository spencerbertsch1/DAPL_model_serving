import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 

# define path to data (pathlib works on any operating system)
PATH_TO_THIS_FILE: Path = Path(__file__).resolve()
ABSPATH_TO_DATA: Path = PATH_TO_THIS_FILE.parent / "titanic.csv"

# read the data into a pandas df - data source: https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html
df = pd.read_csv(str(ABSPATH_TO_DATA), sep=',')
sorted_ages: list = list(df['Age'].unique())  # <-- we use this later on 
sorted_fares: list = list(df['Fare'].unique())  # <-- we use this later on 
sorted_ages.sort()
sorted_fares.sort()
# replace the string representation of sex with a binary int
df['Encoded_sex'] = np.where(df['Sex']=='male', 1, 0)
# define the predictor and response variables
X = df[['Pclass', 'Encoded_sex', 'Age', 'Fare']]
y = df['Survived']

#  define training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=11)

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100) 

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
# at this point we could use some hyperparameter optimization to improve performance, but we can skip that

# re-train on the entire data set for the model that we will use for deployment
clf.fit(X, y)

# grab the feature importance vector from the trained model 
feature_imp = pd.Series(clf.feature_importances_, index = X.columns).sort_values(ascending = False)

# ------------------------------------- STREAMLIT CODE -------------------------------------
st.markdown(''' # Titanic Survival Machine Learning Dashboard ''')
st.image('https://i.pinimg.com/736x/e6/5c/b3/e65cb36df699c19ed3f08f3f78ae94d3--titanic-messages.jpg')

cabin_class = st.selectbox(
     'What is your cabin class?',
     (1, 2, 3))

sex_encoding: dict = {'Male': 1, 'Female': 0}
sex = st.selectbox(
     'What is your sex?',
     ('Male', 'Female'))

age = st.select_slider(
     'How old are you?',
     options=sorted_ages)

fare = st.select_slider(
     'How much did you pay for your ticket?',
     options=sorted_fares)

prediction = round(clf.predict_proba([[cabin_class, sex_encoding[sex], age, fare]])[0][1], 4)

if st.button('Run Machine Learning Model'):
     st.write(f'Probability of survival: {prediction*100}%')

st.markdown(''' ### Let's see how well our model performed ''')
st.markdown(f''' Model Accuracy: **{round(acc, 3)*100}%** ''')
st.markdown(f''' Model F1 Score: **{round(f1, 3)*100}%** ''')

st.markdown(''' ### Relative Feature Importance in Random Forest Model: ''')
st.bar_chart(feature_imp.sort_values(ascending=True))

st.markdown(''' ##### Image Source: [Pinterest](https://www.pinterest.com/pin/487373990908858394/) ''')

# this dashboard reads the entire data set and re-trains models every time an inference happens. This is usually 
# very bad practice, but it's ok because there are only 800 samples and 4 features. The keen reader could fix this 
# problem by loading the trained model from disk instead of re-training it every time - that would only require a few lines of code. 
