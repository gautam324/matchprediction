import streamlit as st
import pickle
import pandas as pd

# Define teams and cities
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load the model
pipe = pickle.load(open('C:/Users/gauta/Desktop/jupyter projects/ipl match prediction/pipe.pkl', 'rb'))


# Streamlit app
st.title('IPL Win Predictor')

# Sidebar layout
batting_team = st.sidebar.selectbox('Select the batting team', sorted(teams))
bowling_team = st.sidebar.selectbox('Select the bowling team', sorted(teams))
selected_city = st.sidebar.selectbox('Select host city', sorted(cities))
target = st.sidebar.number_input('Target')

# Input data layout
with st.form(key='input_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        score = st.number_input('Score')
    with col2:
        overs = st.number_input('Overs completed')
    with col3:
        wickets = st.number_input('Wickets out')

    submit_button = st.form_submit_button(label='Predict Probability')

# Prediction and display result
if submit_button:
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    win_probability = result[0][1] * 100
    loss_probability = result[0][0] * 100

    st.header(f"{batting_team} - {win_probability:.2f}%")
    st.header(f"{bowling_team} - {loss_probability:.2f}%")
