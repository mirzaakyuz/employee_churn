import streamlit as st
import pandas as pd
import numpy as np
import pickle

from PIL import Image

st.header("Employee Churn Prediction App")

img = Image.open("quit-unsplash_c.jpg")
caption = """
Photo by [Jackson Simmer] 
(https://unsplash.com/@simmerdownjpg) 
on [Unsplash] 
(https://unsplash.com/)"""
st.image(img, width=600)
st.write(caption)


st.write("""
            This app predicts whether the employee give up the position considering given features.
            """)

st.sidebar.header("Please input required data of employee:")


@st.cache
def feature_load():
    with open('features.pkl', 'rb') as m:
        f = pickle.load(m)
        return f


@st.cache(allow_output_mutation=True)
def model_load():
    with open('rf_model_last.pkl', 'rb') as m:
        f = pickle.load(m)
        return f


def user_input_feature():
    satisfaction_level = st.sidebar.slider("Satisfaction level of employee:", 0.0, 1.0, 0.07)
    last_evaluation = st.sidebar.slider("Last evaluation result:", 0.0, 1.0, 0.07)
    number_project = st.sidebar.slider("Number of project involved:", 0, 7, 7)
    average_monthly_hours = st.sidebar.slider("Average monthly hour:", 96, 310, 280, step=1)
    time_spent_company = st.sidebar.selectbox("How long work employee?", ("2", "3", "4", "5", "6", "7+"))
    work_accident = st.sidebar.radio("Has the employee ever had a work accident?", ("Yes", "No"))
    department = st.sidebar.selectbox("Which department work employee for ?",
                                      ("sales", "accounting", "hr", "technical", "support", "management",
                                       "IT", "product_mng", "marketing", "RandD"))
    promotion_last_5years = st.sidebar.radio("Has the employee promoted in las 5 years?", ("Yes", "No"))
    salary = st.sidebar.radio("Salary level of employee", ("low", "medium", "high"))

    data = {
        "satisfaction_level": satisfaction_level,
        "last_evaluation": last_evaluation,
        "number_project": number_project,
        "average_montly_hours": average_monthly_hours,
        "time_spend_company": time_spent_company,
        "Work_accident": work_accident,
        "promotion_last_5years": promotion_last_5years,
        "department": department,
        "salary": salary
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_feature()
st.write("Below is your selection:")
input_show = pd.DataFrame(input_df.values, columns=["Satisfaction level",
                                                    "Last evaluation grade",
                                                    "Number of Project",
                                                    "Monthly working hours",
                                                    "Tenure",
                                                    "Had work accident?",
                                                    "Department",
                                                    "Has promoted before?",
                                                    "Salary level"])

st.write(input_show.T)


def workload(x):
    if x <= 220:
        return 'moderate'
    else:
        return 'over'


def seven_plus(x):
    l = ['7', '8', '10']
    if x in l:
        return '7+'
    else:
        return x


input_df['time_cat'] = input_df['time_spend_company'].apply(seven_plus)
input_df['workload'] = input_df['average_montly_hours'].apply(workload)

features = feature_load()

df = pd.get_dummies(input_df).reindex(columns=features, fill_value=0)

df.drop('left', axis=1, inplace=True)

# st.write(df.columns)
model = model_load()

st.write("If your selection is done, please Click on Predict button.")
if st.button("Predict"):
    prediction = model.predict(df)
    pred_probability = model.predict_proba(df)

    if prediction == 0:
        st.success(f"The employee with a probability of % {pred_probability.flat[0]:.3f} would stay with us.")
    else:
        st.error(f"The employee with a probability of % {pred_probability.flat[1]:.3f}  may resign from job.")



