import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv('Cholera outbreak dataset.csv')

# columns to drop: Test_results, Date, Outcome
X = df.drop(columns=["Outcome","Test_date","Test_result"])

y=df['Outcome']
X=X.to_numpy()
y=y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)

# Title of the app
st.title('Cholera Status Prediction App')

# Input features from the user
sex = st.selectbox('Sex', [1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
age = st.slider('Age', 0, 100, 25)
subcounty = st.selectbox('Subcounty', [1, 2, 3, 4])
water = st.selectbox('Water Source Quality', [1, 2, 3, 4])
sanitation = st.selectbox('Sanitation Status', [1, 2, 3, 4])
income = st.selectbox('Income Level', [1, 2, 3, 4, 5])
informal_settlement = st.selectbox('Lives in Informal Settlement', [1, 2])

# Predict button
if st.button('Predict Cholera Status'):
    # Create a dataframe for the input features
    input_data = pd.DataFrame({
        'Sex': [sex],
        'Age': [age],
        'Subcounty': [subcounty],
        'Water': [water],
        'Sanitation': [sanitation],
        'Income': [income],
        'Informal_settlement': [informal_settlement]
    })
    
    # Load the trained model and scaler (assumed to be saved previously)
    scaler = StandardScaler()
    #model = RandomForestClassifier(random_state=42)
    
    #model = LogisticRegression(class_weight='balanced')
    #model = SVC(C=10,gamma=0.001,kernel='rbf',class_weight='balanced')
    model = LogisticRegression(solver='newton-cg',penalty='l2', max_iter=10000,class_weight='balanced')



    # feature scaling
    X_train_trans = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model fitting
    model.fit(X_train_trans, y_train)

    preds = model.predict(X_test_scaled)

    print(confusion_matrix(y_test, preds))

    # Scaling the input data
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Making the prediction
    prediction = model.predict(input_data_scaled)
    
    st.write(confusion_matrix(y_test, preds))
    st.write("Accuracy: ", accuracy_score(y_test, preds))
    #st.write(classification_report(y_test, preds))
    # Displaying the result
    if prediction[0] == 1:
        st.write('The model predicts that this individual has cholera.')
    else:
        st.write('The model predicts that this individual does not have cholera.')
