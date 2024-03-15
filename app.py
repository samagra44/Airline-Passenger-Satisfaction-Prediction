import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk

# loading the model
with open("saved_model.pkl",'rb') as model_file:
    rfc = pk.load(model_file)

st.set_page_config(
    page_title="Airline Passenger Satisfaction Prediction",
    layout="centered",
    page_icon="bar-chart"
)

# Label Encoding of features
satisfaction_labels = {'satisfied':1, 'dissatisfied':0}
Customer_Type_lables = {'Loyal Customer':0, 'disloyal Customer':1}
Type_of_Travel_labels = {'Personal Travel':1, 'Business travel':0}
Class_labels = {'Eco':1, 'Business':0, 'Eco Plus':2}

# title of the page
st.title("Airline Passenger Satisfaction Prediction 🙋🏻🙎🏻‍♂️")
st.sidebar.title("Airline Passenger Satisfaction Prediction")
st.sidebar.header("Fill the details...")
user_input = {}

# input features

# Numerical Features
user_input_Age = st.sidebar.slider("Age",min_value=10, max_value=100, value=20)
user_input_Flight_Distance = st.sidebar.slider("Flight_Distance",min_value=100, max_value=5000, value=50)
user_input_Seat_comfort = st.sidebar.slider("Seat comfort",min_value=0, max_value=10, value=3)
user_input_Departure_Arrival_time_convenient = st.sidebar.slider("Departure/Arrival time convenient",min_value=0, max_value=10, value=2)
user_input_Food_and_drink = st.sidebar.slider("Food and drink",min_value=0, max_value=10, value=4)
user_input_Gate_location = st.sidebar.slider("Gate location",min_value=0, max_value=10, value=20)
user_input_Inflight_wifi_service = st.sidebar.slider("Inflight wifi service",min_value=0, max_value=10, value=7)
user_input_Inflight_entertainment = st.sidebar.slider("Inflight entertainment",min_value=0, max_value=10, value=5)
user_input_Online_support = st.sidebar.slider("Online support",min_value=0, max_value=10, value=6)
user_input_Ease_of_Online_booking = st.sidebar.slider("Ease of Online booking",min_value=0, max_value=10, value=4)
user_input_On_board_service = st.sidebar.slider("On-board service",min_value=0, max_value=10, value=8)
user_input_Leg_room_service = st.sidebar.slider("Leg room service",min_value=0, max_value=10, value=3)
user_input_Baggage_handling = st.sidebar.slider("Baggage handling",min_value=0, max_value=10, value=5)
user_input_Checkin_service = st.sidebar.slider("Checkin service",min_value=0, max_value=10, value=6)
user_input_Cleanliness = st.sidebar.slider("Cleanliness",min_value=0, max_value=10, value=5)
user_input_Online_boarding = st.sidebar.slider("Online boarding",min_value=0, max_value=10, value=6)
user_input_Departure_Delay_in_Minutes = st.sidebar.slider("Departure Delay in Minutes",min_value=0, max_value=500, value=100)
user_input_Arrival_Delay_in_Minutes = st.sidebar.slider("Arrival Delay in Minutes",min_value=0, max_value=10, value=7)

# Categorical Fetures
user_input_Type_of_Travel = st.sidebar.selectbox("Type of Travel",list(Type_of_Travel_labels.keys()))
user_input_Customer_Type = st.sidebar.selectbox("Customer Type",list(Customer_Type_lables.keys()))
user_input_Class_labels = st.sidebar.selectbox("Class",list(Class_labels.keys()))

# function to make prediction
def make_prediction():
    input_array = np.array([[Customer_Type_lables[user_input_Customer_Type],
                                user_input_Age,
                                Type_of_Travel_labels[user_input_Type_of_Travel],
                                Class_labels[user_input_Class_labels],
                                user_input_Flight_Distance,
                                user_input_Seat_comfort,
                                user_input_Departure_Arrival_time_convenient,
                                user_input_Food_and_drink,
                                user_input_Gate_location,
                                user_input_Inflight_wifi_service,
                                user_input_Inflight_entertainment,
                                user_input_Online_support,
                                user_input_Ease_of_Online_booking,
                                user_input_On_board_service,
                                user_input_Leg_room_service,
                                user_input_Baggage_handling,
                                user_input_Checkin_service,
                                user_input_Cleanliness,
                                user_input_Online_boarding,
                                user_input_Departure_Delay_in_Minutes,
                                user_input_Arrival_Delay_in_Minutes
                                ]])
    # making prediction from input features
    # here I am using Random Forest Classifier Model
    prediction = rfc.predict(input_array)
    # Displaying the model prediction
    st.write("Model Prediction", prediction[0])

    # checking condition and displaying the result.
    if prediction[0] == 0:
        st.error("Passenger Not Satisfied ❌🙆")
    else:
        st.success("Passenger is Satisfied ✅👌")

# display result when submit button is clicked.
if st.sidebar.button("Submit"):
    with st.spinner("Predicting ⏳..."):
        make_prediction()