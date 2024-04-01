# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 21:05:52 2022

@author: USER
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

st.title('Bondora Churn Prediction')


st.sidebar.header('User Input Parameters')

def user_input_features():
    
    country = st.sidebar.selectbox("Country",("EE","FI","ES"))
    ms = st.sidebar.selectbox("MaritalStatus",("Not_specified","Single","Married","Cohabitant","Divorced","Widow"))
    es = st.sidebar.selectbox("EmploymentStatus",("Not_specified","Fully-Employed","Entrepreneur","Retiree",
                                                  "Self-Employed","Partially-Employed"))
    
    use = st.sidebar.selectbox("UseOfLoan",("Not Set","Other ","Home Improvement","Loan Consolidation",
                                            "Vehicle","Business","Travel","Health","Real Estate","Education",
                                            "Purchase of Machinery Equipment","Other Business","Accounts Receivable Financing",
                                            "Working Capital Financing","Acquisition of Stocks","Acquisition of Real Estate",
                                            "Construction Finance"))
    new = st.sidebar.selectbox("NewCreditCustomer",("True","False"))
    ld = st.slider("LoanDuration",min_value=0,max_value=65,step=1)
    
    am = st.sidebar.number_input("Amount",min_value=0,max_value=11000,value=0)
    inte = st.sidebar.number_input("Interest",min_value=0,max_value=300,value=0)
    pb = st.sidebar.number_input("PrincipalBalance", min_value=-35, max_value=10600,value=0)
    mp = st.sidebar.number_input("MonthlyPayment",min_value=0,max_value=2400,step=0)
    ipb = st.sidebar.number_input("InterestAndPenaltyBalance",min_value=-2,max_value=65000,step=0)
    prbl = st.sidebar.number_input("PreviousRepaymentsBeforeLoan",min_value=0,max_value=34000,step=0)
    dti = st.sidebar.number_input("DebtToIncome",min_value=0,max_value=100,step=0)
    
    
    new = {
         'Country': country,
         'MaritalStatus': ms,
         'EmploymentStatus': es,
         'UseOfLoan': use,
         'NewCreditCustomer': new,
         'Amount': am,
         'Interest': inte,
         "LoanDuration": ld,
         "MonthlyPayment": mp,
         "DebtToIncome": dti,
         "PrincipalBalance": pb,
         "InterestAndPenaltyBalance": ipb,
         "PreviousRepaymentsBeforeLoan": prbl
            }
    features = pd.DataFrame(new,index = [0])
    return features 
    
df = user_input_features()
st.write(df)



import pickle

with open(file="final_model.pkl",mode="rb") as f:
    model = pickle.load(f)


st.write("Model loaded")



result = model.predict(df)
st.subheader('Predicted Result')

if result[0]==0:
    st.write("Customer will not Churn")
    
else:
    st.write("Customer will Churn")