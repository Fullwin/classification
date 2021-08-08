import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
## `Stroke Prediction`

This app predicts if you are vulnerable of a stroke based on multiple variables.

""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
Data Source: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        gender = st.sidebar.selectbox('Gender',('Male','Female'))
        age = st.sidebar.slider('Age', 1,100)
        hypertension = st.sidebar.selectbox('Hypertension (0 = No, 1 = Yes)',(0,1))
        heart_disease = st.sidebar.selectbox('Heart Disease (0 = No, 1 = Yes)',(0,1))
        ever_married = st.sidebar.selectbox('Ever married?',('Yes','No'))
        work_type = st.sidebar.selectbox('Work Type',('Private','Self-employed','Govt_job','children','Never_worked'))
        Residence_type = st.sidebar.selectbox('Resident Type',('Urban','Rural'))
        avg_glucose_level = st.sidebar.slider('Average Gluclose Level', 50,300)
        bmi = st.sidebar.slider('BMI', 10,100)
        smoking_status = st.sidebar.selectbox('Smoking Status',('never smoked','formerly smoked','smokes','Unknown'))
        data = {'gender': gender,
                'age': age,
                'hypertension' : hypertension,
                'heart_disease' : heart_disease,
                'ever_married' : ever_married,
                'work_type' : work_type,
                'Residence_type' : Residence_type,
                'avg_glucose_level' : avg_glucose_level,
                'bmi' : bmi,
                'smoking_status' : smoking_status,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
strokes_raw = pd.read_csv('stroke_cleaned.csv')
stroke = strokes_raw.drop(columns=['stroke'])
df = pd.concat([input_df,stroke],axis=0)


#Converting categorical to numerical 
encode = ['gender','ever_married','work_type','Residence_type','smoking_status']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col) #Convert categorical varibales to dummy numbers
    df = pd.concat([df,dummy], axis=1) #Add them to the dataframe
    del df[col] #Delete old categorical columns

if uploaded_file is not None:
    df = df[:len(input_df.index)]
else:
    df = df[:1] #Select first row of user input only

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('stroke_clf.pkl', 'rb'))


# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
Stroke_Result = np.array(['No','Yes'])
st.write(Stroke_Result[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
