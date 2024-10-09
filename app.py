import streamlit as st
import sklearn
import pickle

import pandas as pd
import numpy as np
import pickle

from datetime import datetime
import time
import os
from dotenv import load_dotenv
import google.generativeai as ggi

st.set_page_config(page_title="SmartGrade", page_icon=":book:")

load_dotenv(".env")
fetched_api_key = st.secrets["API_KEY"]
ggi.configure(api_key=fetched_api_key)
llm_model = ggi.GenerativeModel("gemini-pro")


jamb_model = pickle.load(open('jamb_model.pkl', 'rb'))
waec_model = pickle.load(open('waec_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

full_data = pd.read_csv("Cleaned Academic Survey.csv")
data = full_data[(full_data["Class"]=="Graduate")]


out_school = list(data["Outside School Study Hours"].unique())
method_of_exam_preparation = list(data["Method of Exam Preparation"].unique())
family_income = list(data["Family's Monthly Income"].unique())
stress = list(data["Sources of Stress"].unique())
quality_of_teaching = list(data["Quality of Teaching"].unique())
learning_style = list(data["Learning Style"].unique())
location = list(data["Location"].unique())
problem_solving = list(data["Ability to solve problems"].unique())
academic_motivation = list(data["Academic Motivation"].unique())
daily_study_hours = list(data["Daily Study Hours"].unique())
learning_environment = list(data["Learning Environment"].unique())
performance_feedback = list(data["Academic Performance Feedback"].unique())
extra_lessons = list(data["Number of extra lessons attended"].unique())
type_of_school = list(data["Type of School"].unique())
extracurricular = list(data["Extracurricular activities"].unique())
learning_resources = list(data["Availability of Learning Resources"].unique())
class_type = list(data["Class Type"].unique())
intelligence = list(data["Intelligence Rating"].unique())
age = list(data["Age"].unique())
preparatory_cbt = list(data["Number of preparatory CBT taken"].unique())
ability_confidence = list(data["Confidence in Ability"].unique())
financial_support = list(data["Financial Support"].unique())
gender = list(data["Gender"].unique())
nature_of_school = list(data["Nature of School"].unique())


def main():
    st.title("SmartGrade")
    st.image("books.jpg")
    st.subheader("A Grade Prediction System")

    st.markdown("**Hi, Scholar!, Welcome to your personalized grade prediction app, designed to help you take control of your academic journey. Using advanced algorithms, this tool analyzes your performance to predict your final exam grades, giving you valuable insights into your strengths and areas for improvement.**")
    st.sidebar.markdown("""
   Are you a final year secondary school student preparing for your final exams (WASSCE, JAMB, and NECO), SmartGrade is here for you! Introducing our innovative web app designed to empower students by predicting their grades in final exams with precision. Tailored specifically for secondary school students, the app leverages data-driven algorithms to analyze performance trends, offering personalized grade forecasts. This tool helps students set realistic goals, identify areas needing improvement, and better prepare for their final exams, ensuring they stay on track for academic success.
    """)
    st.sidebar.markdown("**Built by The Data Squad**")
    
    #input prompts
    st.write(" ")
    st.write(" ")
    st.markdown("### Get Started!")
    st.write('----')
    st.markdown("*To get started, please answer the following questions;*")
    col1, col2 = st.columns([1,1])
    with col1:
        age_opt = st.radio("What is your age range?", age)
    with col2:
        gender_opt = st.radio("Select your gender", gender)

    col3, col4 = st.columns([1,1])
    with col3:
        class_type_opt = st.radio("Which class are you in?", class_type)
    with col4:
        location_opt = st.radio("Where is your school located?", location)
    
    col5, col6 = st.columns([1,1])
    with col5:
        type_of_school_opt = st.radio("Is your school boarding or day?", type_of_school)
    with col6:
        nature_of_school_opt = st.radio("Is your school private or public?", nature_of_school)
    
    col7, col8 = st.columns([1,1])
    with col7:
        quality_of_teaching_opt = st.radio("How would you rate the quality of teaching in your school?", quality_of_teaching)
    with col8:
        performance_feedback_opt = st.radio("How often do you receive feedback on your academic performance?", performance_feedback)

    col9, col10 = st.columns([1,1])
    with col9:
        learning_resources_opt  = st.radio("Are learning resources (books, internet, laboratories, and tutorials) readily available to you?", learning_resources)
    with col10:
        learning_environment_opt = st.radio("How comfortable is your classroom environment for learning on a scale of 1-5?", ["5", "4", "3", "2", "1"])
    
    col11, col12 = st.columns([1,1])
    with col11:
        family_income_opt = st.radio("What is your family's monthly income range?", family_income)
    with col12:
        financial_support_opt = st.radio("Do you receive financial support for your education? (e.g., scholarships, loans)", financial_support)    
    col13, col14 = st.columns([1,1])
    with col13:
        out_school_opt = st.radio("How many hours do you study per day outside of school hours?", out_school)
    with col14:
        extracurricular_opt = st.radio("Do you participate in extracurricular activities", extracurricular)
    col15, col16 = st.columns([1,1])
    with col15:
        academic_motivation_opt = st.radio("How would you rate your motivation to perform well academically?", academic_motivation)
    with col16:
        stress_opt = st.radio("What is your primary source of stress?", ["Academic Pressure", "Family expectations", "Financial issues", "Social relationships", "None"])
    col17, col18 = st.columns([1,1])
    with col17:
        inteligence_opt = st.radio("How would you rate your overall intelligence based on past performances?", intelligence)
    with col18:
        ability_confidence_opt = st.radio("How confident are you in your ability to perform well in your final exams?", ability_confidence)
    col19, col20 = st.columns([1,1])
    with col19:
        problem_solving_opt = st.radio("How often do you solve problems?", problem_solving)
    with col20:
        learning_style_opt = st.radio("What is your preferred learning style?", ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"])
    col21, col22 = st.columns([1,1])
    with col21:
        preparatory_cbt_opt = st.radio("How many preparatory CBTs have you taken?", ["1-2", "None", "3-5", "More than 5"])
    with col22:
        extra_lessons_opt = st.radio("How often do you take extra lessons in a week?", extra_lessons)
    col23, col24 = st.columns([1,1])
    with col23:
        method_of_exam_preparation_opt = st.radio("How do you prefer to prepare for exams?", method_of_exam_preparation)
    with col24:
        daily_study_hours_opt = st.radio("How many hours do you study per day?", daily_study_hours)
    st.write('----')

    labeled_categories = np.array(encoder.fit_transform([age_opt, gender_opt, class_type_opt, location_opt, type_of_school_opt, nature_of_school_opt,
    quality_of_teaching_opt, performance_feedback_opt, learning_resources_opt, learning_environment_opt, family_income_opt, financial_support_opt,
    out_school_opt, extracurricular_opt, academic_motivation_opt, stress_opt, inteligence_opt, ability_confidence_opt, problem_solving_opt, learning_style_opt, preparatory_cbt_opt,
    extra_lessons_opt, method_of_exam_preparation_opt, daily_study_hours_opt])).reshape(1, -1)

    #processed_categories = np.concatenate([labeled_categories, numerical_features], axis=1).reshape(1, -1)
    scaled_features = scaler.transform(labeled_categories)

    def jamb_predict(model, features):
        pred = model.predict_proba(features)
        prob_list =[]
        for prob in pred[:, :]:
            prob_list.append(prob * 100)
            result = list(prob_list[0])
            results_df = pd.DataFrame({"Score Range":["Less than 200", "Between 200 and 270", "270 and above"], "Probability(%)": result})
            return results_df

    def waec_predict(model, features):
        pred = model.predict_proba(features)
        prob_list =[]
        for prob in pred[:, :]:
            prob_list.append(prob * 100)
            result = list(prob_list[0])
            results_df = pd.DataFrame({"Grade":["5 credits and above", "Less than 5 credits"], "Probability(%)": result})
            return results_df

    jamb_prediction = jamb_predict(jamb_model, scaled_features)
    waec_prediction = waec_predict(waec_model, scaled_features)

    jamb_pred = ""
    if jamb_model.predict(scaled_features) == 0:
        jamb_pred = "less than 200"
    elif jamb_model.predict(scaled_features) == 1:
        jamb_pred = "betwween 200 and 270"
    elif jamb_model.predict(scaled_features) == 2:
        jamb_pred = "above 270"

    waec_pred = ""
    if waec_model.predict(scaled_features) == 0:
        jamb_pred = "5 credits and above"
    elif waec_model.predict(scaled_features) == 1:
        jamb_pred = "less than 5 credits"

    prompt = f"This grade prediction model says my JAMB score will be {jamb_pred}, it also said I'll have {waec_pred} in my upcoming WASSCE. Any tips for me as a final year secondary school student to make me pass my final exams exceptionally well. Start with; Hello Scholar, ..."
    response = llm_model.generate_content(prompt, stream=True)
    response.resolve()


    col25,col26 = st.columns([1,1])
    with col25:
        if st.button('Predict your JAMB score'):
            st.success("Here are your predictions")
            st.dataframe(jamb_prediction, hide_index=True)
    with col26: 
        if st.button('Predict your WAEC score'):
            st.success("Here are your predictions")
            st.dataframe(waec_prediction, hide_index=True)


    if st.button("Get tips on acing your final exams"):
        st.markdown(response.text)

if __name__ == "__main__":
    main()

