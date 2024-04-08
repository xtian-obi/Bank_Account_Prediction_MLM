import pandas as pd
import streamlit as st

#importing the dataframe
df1 = pd.read_csv('Financial_inclusion_dataset.csv')
df=pd.read_csv('Financial_inclusion_dataset.csv')

#Information on all columns
print(df.info())

df.head(1)

df.drop(['country', 'year', 'uniqueid'],axis=1,inplace=True)

print(df.head(2))

from sklearn.preprocessing import LabelEncoder

culms= ['location_type', 'cellphone_access', 'gender_of_respondent', 'relationship_with_head','education_level', 'marital_status', 'job_type']

lb=LabelEncoder()

encoder={}
for i in culms:
    lb=LabelEncoder()
    df[i]=lb.fit_transform(df[i])
    encoder[i]=lb

from sklearn.ensemble import RandomForestClassifier

rc=RandomForestClassifier(n_estimators=200)

from sklearn.model_selection import train_test_split

X=df.drop('bank_account',axis=1)
y=df['bank_account']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

rc.fit(X_train,y_train)

pred=rc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))

with st.form('Enter basic details Below'):
    location_type=st.selectbox('Enter your location',df1.location_type.unique())
    cellphone_access=st.selectbox('Enter your Cellphone access',df1.cellphone_access.unique())
    household_size = st.number_input('Enter the size of your household')
    age=st.number_input('Enter the age of your')
    gender=st.selectbox('Enter your Gender',df1.gender_of_respondent.unique())
    relationship=st.selectbox('Enter your relationship with head',df1.relationship_with_head.unique())
    education=st.selectbox('Enter your Education status',df1.education_level.unique())
    job_type=st.selectbox('Enter your Job type',df1.job_type.unique())
    marital_status=st.selectbox('Enter your marital status',df1.marital_status.unique())
    submit = st.form_submit_button('Click Here to Predict Likeihood of Bank Account')

key=['location_type', 'cellphone_access', 'gender_of_respondent', 'relationship_with_head','education_level','job_type','marital_status']
value=[location_type, cellphone_access, gender,relationship,education,job_type,marital_status]

user_input=dict(zip(key,value))

if submit:
    for i,j in encoder.items():
        user_input[i]=j.transform([user_input[i]])[0]
    features=[value for value in user_input.values()]
    features.insert(2,household_size)
    features.insert(3, age)
    st.write('The Answer is', rc.predict([features])[0])
