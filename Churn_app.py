import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import os
from PIL import Image
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#Function to load the uploaded data by the client
def loadData(file):
     df = pd.read_csv(file)
     return df
     
#Funcion that performs preprocessing steps
def preprocessing(df):
     df1 = df[['gender', 'SeniorCitizen', 'PaymentMethod', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'MonthlyCharges','tenure', 'TotalCharges', 'Churn']].copy()
     df1['TotalCharges']=pd.to_numeric(df1['TotalCharges'],errors='coerce')
     df1['tenure']=pd.to_numeric(df1['tenure'],errors='coerce')
     df1 = df1.dropna()

     #Create dummy variables for the categorical columns
     encode = ['gender','PaymentMethod','Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']
     
     for col in encode:

          dummy = pd.get_dummies(df1[col], prefix=col)

          df1 = pd.concat([df1,dummy], axis=1)

          del df1[col]
    
     df1['Churn'] = np.where(df1['Churn']=='Yes', 1, 0)
     return df1

#Random Forest Classifier
def RandomForest(X_train, X_test, y_train, y_test):
	# Train the model
     RF = RandomForestClassifier(random_state=12,)
     RF.fit(X_train, y_train)
     pred = RF.predict(X_test)
     score = metrics.accuracy_score(y_test, pred)
     report = classification_report(y_test, pred)
     conf_mat = confusion_matrix(y_test, pred)
     return score, report, conf_mat, RF

#K Nearest Neighbours Classifier
def Knn_Classifier(X_train, X_test, y_train, y_test):
     clf = KNeighborsClassifier(n_neighbors=7)
     clf.fit(X_train, y_train)
     pred = clf.predict(X_test)
     score = metrics.accuracy_score(y_test, pred)
     report = classification_report(y_test, pred)
     conf_mat = confusion_matrix(y_test, pred)
     return score, report, conf_mat, clf

#Logistic Regression
def LogReg(X_train, X_test, y_train, y_test):
     logreg = LogisticRegression()
     logreg.fit(X_train, y_train)
     pred = logreg.predict(X_test)
     score = metrics.accuracy_score(y_test, pred)
     conf_mat = confusion_matrix(y_test, pred) 
     return score, conf_mat, logreg

#Function that allows the user to enter his own input and return the input in a dataframe 
def accept_user_data():
     gender = st.selectbox('Gender:',('Female', 'Male'))
     SeniorCitizen = st.selectbox('SeniorCitizen:', (1,0))
     dependents = st.selectbox('Dependent:', ('Yes', 'No'))
     partner = st.selectbox('Partner', ('Yes', 'No'))
     tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
     contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
     paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
     PaymentMethod = st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
     monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
     totalcharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)
     DeviceProtection = st.selectbox('DeviceProtection:', ('Yes', 'No') )
     mutliplelines = st.selectbox("Does the customer have multiple lines",('Yes','No','No phone service'))
     phoneservice = st.selectbox('Phone Service:', ('Yes', 'No'))
     internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
     onlinesecurity = st.selectbox("Does the customer have online security",('Yes','No','No internet service'))
     onlinebackup = st.selectbox("Does the customer have online backup",('Yes','No','No internet service'))
     techsupport = st.selectbox("Does the customer have technology support", ('Yes','No','No internet service'))
     streamingtv = st.selectbox("Does the customer stream TV", ('Yes','No','No internet service'))
     streamingmovies = st.selectbox("Does the customer stream movies", ('Yes','No','No internet service'))

     data = {   'gender': gender, 'Dependents': dependents, 'Partner': partner, 'DeviceProtection': DeviceProtection, 'SeniorCitizen':SeniorCitizen,
                'tenure':tenure,
                'PhoneService': phoneservice,
                'MultipleLines': mutliplelines,
                'InternetService': internetservice,
                'OnlineSecurity': onlinesecurity,
                'OnlineBackup': onlinebackup,
                'TechSupport': techsupport,
                'StreamingTV': streamingtv,
                'StreamingMovies': streamingmovies,
                'Contract': contract,
                'PaperlessBilling': paperlessbilling,
                'PaymentMethod':PaymentMethod, 
                'MonthlyCharges': monthlycharges, 
                'TotalCharges': totalcharges
                }
     features_df = pd.DataFrame.from_dict([data])
     return (features_df)
     
#Main
def main():
     st.set_page_config(layout="wide")
     image = Image.open(dname + "/" + "Customer_Churn.png")

     st.image(image)
     st.title("Prediction of Customer Churn")
     st.write("Customer churn is defined as the loss of customers after a certain period of time.")
     
     #Crete a file uploader for csv files only
     file = st.file_uploader("Please choose a csv file", type = ["csv"]) 
     

     st.set_option('deprecation.showPyplotGlobalUse', False)
     if file is not None:
          data_1 = loadData(file)
          data = data_1.copy()

          #Change data types where needed
          data_1['TotalCharges']=pd.to_numeric(data_1['TotalCharges'],errors='coerce')
          data_1['tenure']=pd.to_numeric(data_1['tenure'],errors='coerce')
          data_1= data_1.dropna()

          #Create filters for Payment Method, gender, tenure, MonthlyCharges
          st.sidebar.header('Select what to display')
          payment = data_1['PaymentMethod'].unique().tolist()
          payment_method_selected = st.sidebar.multiselect('Payment Methods', payment, payment)
     
          gender = data_1['gender'].unique().tolist()
          gender_selected = st.sidebar.multiselect('Gender', gender, gender)

          tenure = data_1['tenure']
   
          nb_tenure = st.sidebar.slider("Tenure values", int(tenure.min()), int(tenure.max()), (int(tenure.min()), int(tenure.max())),1)
          
          monthly_charges = data_1['MonthlyCharges']
          nb_monthly_charges = st.sidebar.slider("Monthly Charges", int(monthly_charges.min()), int(monthly_charges.max()), (int(monthly_charges.min()), int(monthly_charges.max())), 1)
     
          mask_payment_method = data_1['PaymentMethod'].isin(payment_method_selected)
          data_filter_1 = data_1[mask_payment_method]
 
          mask_gender = data_filter_1['gender'].isin(gender_selected)
          data_filter_2 = data_filter_1[mask_gender]

          data_filter_3 = data_filter_2[(data_filter_2.tenure >= nb_tenure[0]) & (data_filter_2.tenure <= nb_tenure[1])]
  
          data_filter_4 = data_filter_3[(data_filter_3.MonthlyCharges >= nb_monthly_charges[0]) & (data_filter_3.MonthlyCharges <= nb_monthly_charges[1])]

          #Display the dataframe
          st.write(data_filter_4)
         

          #Preprocess and split the complete dataset in order to prepare for Machine Learning
          data_preprocessed = preprocessing(data)
          train, test = train_test_split(data_preprocessed, test_size = 0.2, random_state = 123)
     
          X_train = train.drop('Churn', axis = 1)
          y_train = train['Churn']
          X_test = test.drop('Churn', axis = 1)
          y_test = test['Churn']

          # Visualization Section
          choose_viz = st.sidebar.selectbox("Choose the visualization:",
		["None", "Churn Rate Distribution","Categorical Features and Churn", "Boxplots Numerical Features", "Pairplots Numerical Features"])
          #Visualize Churn Rate Distribution
          if(choose_viz == "Churn Rate Distribution"):
               fig = plt.figure(figsize=(3, 5))
               churn_counts = data_filter_4.Churn.value_counts()
               df_value_counts = pd.DataFrame(churn_counts)
               fig = px.bar(df_value_counts, y=df_value_counts["Churn"], title="Churn Rate Distribution")
               st.plotly_chart(fig, use_container_width = True)

          #Visualize countplots for categorical features
          elif(choose_viz == "Categorical Features and Churn"):
               cat_vars = ['gender', 'SeniorCitizen', 'PaymentMethod', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']
               fig = plt.figure(figsize=(3, 5))
               
               col1, col2 = st.columns(2)
               flag = True
               for var in (cat_vars):     
                    fig = px.histogram(data_filter_4, x = var, color= "Churn")
                    if flag == True:
                         col1.plotly_chart(fig, use_container_width = True)
                         flag = False
                    else:
                         col2.plotly_chart(fig, use_container_width = True)
                         flag = True 
          #Visualize boxplots for numerical features
          elif(choose_viz == "Boxplots Numerical Features"):
               col1, col2 = st.columns(2)
               fig = plt.figure(figsize=(3,5))
               fig = px.box(data_filter_4, x= data_filter_4['Churn'], y=data_filter_4["MonthlyCharges"])
          
               col1.plotly_chart(fig, use_container_width = True)
               fig = px.box(data_filter_4, x= data_filter_4['Churn'], y=data_filter_4["TotalCharges"])
               col2.plotly_chart(fig, use_container_width = True)
               fig = px.box(data_filter_4, x= data_filter_4['Churn'], y=data_filter_4["tenure"])
               col1.plotly_chart(fig, use_container_width = True)

          #Visualize pairplots for numerical features
          elif(choose_viz == "Pairplots Numerical Features"):
               fig = sns.pairplot(data_filter_4[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']], hue='Churn', plot_kws=dict(alpha=.3, edgecolor='none'), height=2, aspect=1.1)
               st.pyplot(fig)
           
          #Machine Learning Models
          choose_model = st.sidebar.selectbox("Choose the Machine Learning Model",
		["None","Random Forest Classifier", "K-Nearest Neighbours", "Logistic Regression"])
     
          #Accuracy with Random Forest Classifier
          if(choose_model == "Random Forest Classifier"):
          
               score, report, conf, model = RandomForest(X_train, X_test, y_train, y_test)

               #Show the accuracy score          
               st.text("Accuracy score test with Random Forest Classifier is: ")
               score = score*100
               res = "{:.2f}".format(score)
               st.write(res, "%")
               
               #User's input prediction
               try:
                    if(st.checkbox("Do you wish to predict on your own Input?")):
                         user_prediction_data = accept_user_data()
                    
                         encode = ['gender','PaymentMethod','Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']
    
                         for col in encode:

                              dummy = pd.get_dummies(user_prediction_data[col], prefix=col)

                              user_prediction_data= pd.concat([user_prediction_data,dummy], axis=1)

                              del user_prediction_data[col]
                     
                         st.write("Your input is:")
                         st.write(user_prediction_data)
                         
                         missing_cols = set( X_train.columns ) - set( user_prediction_data.columns )
                         # Add a missing column in test set with default value equal to 0
                         for c in missing_cols:
                              user_prediction_data[c] = 0
                         # Ensure the order of column in the test set is in the same order than in train set
                         user_prediction_data = user_prediction_data[X_train.columns]
               
                         pred = model.predict(user_prediction_data)
                         st.write("Your prediction output is:")
                         if pred == 1:
                              st.write("The customer will churn")
                         else:
                              st.write("The customer will not churn")
               except:
                    pass

          #Accuracy with K-Nearest Neighbours
          if(choose_model == "K-Nearest Neighbours"):
               score, report, conf, clf = Knn_Classifier(X_train, X_test, y_train, y_test)

               #show the accuracy score
               st.text("Accuracy score test with K-Nearest Neighbours model is: ")
               score = score*100
               res = "{:.2f}".format(score)
               st.write(res, "%")
             
               #User's input prediction
               try:
                    if(st.checkbox("Do you wish to predict on your own Input?")):
                         user_prediction_data = accept_user_data()
                         
                         encode = ['gender','PaymentMethod','Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']
    
                         for col in encode:

                              dummy = pd.get_dummies(user_prediction_data[col], prefix=col)

                              user_prediction_data= pd.concat([user_prediction_data,dummy], axis=1)

                              del user_prediction_data[col]

                         st.write("Your input is:")
                         st.write(user_prediction_data)
                         missing_cols = set( X_train.columns ) - set( user_prediction_data.columns )
                         # Add a missing column in test set with default value equal to 0
                         for c in missing_cols:
                              user_prediction_data[c] = 0
                         # Ensure the order of column in the test set is in the same order than in train set
                         user_prediction_data = user_prediction_data[X_train.columns]
               
                         pred = clf.predict(user_prediction_data)
                         st.write("Your prediction output is:")
                         if pred == 1:
                              st.write("The customer will churn")
                         else:
                              st.write("The customer will not churn")
               except:
                    pass
          
          #Accuracy with Logistic Regression
          if(choose_model == "Logistic Regression"):
               score, conf, logreg = LogReg(X_train, X_test, y_train, y_test)
               
               #show the accuracy score
               st.text("Accuracy score test with Logistic Regression model is: ")
               score = score*100
               res = "{:.2f}".format(score)
               st.write(res, "%")
               
               #User's input prediction
               try:
                    if(st.checkbox("Do you wish to predict on your own Input?")):
                         user_prediction_data = accept_user_data()
                         #st.write(user_prediction_data)
                         encode = ['gender','PaymentMethod','Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']
    
                         for col in encode:

                              dummy = pd.get_dummies(user_prediction_data[col], prefix=col)

                              user_prediction_data= pd.concat([user_prediction_data,dummy], axis=1)

                              del user_prediction_data[col]
                         
                         st.write("Your input is:")
                         st.write(user_prediction_data)
                         missing_cols = set( X_train.columns ) - set( user_prediction_data.columns )
                         # Add a missing column in test set with default value equal to 0
                         for c in missing_cols:
                              user_prediction_data[c] = 0
                         # Ensure the order of column in the test set is in the same order than in train set
                         user_prediction_data = user_prediction_data[X_train.columns]
                        
                         pred = logreg.predict(user_prediction_data)
                         st.write("Your prediction output is:")
                         if pred == 1:
                              st.write("The customer will churn")
                         else:
                              st.write("The customer will not churn") 
               except:
                    pass
         
          
if __name__ == "__main__":
	main()
