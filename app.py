import streamlit as st
import pickle
import pandas as pd

# Teams and cities data
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai',
          'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur',
          'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

# Load pre-trained pipeline model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Streamlit app title
st.title('IPL Win Predictor')

# Input fields for teams
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Input field for city
selected_city = st.selectbox('Select host city', sorted(cities))

# Input field for target
target = st.number_input('Target', min_value=0)

# Input fields for score, overs, and wickets
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Prediction button
if st.button('Predict Probability'):
    # Compute derived features
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0  # Handle division by zero
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0  # Handle division by zero

    # Create input DataFrame for model
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balla_left': [balls_left],
        'wickets_left': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    try:
        # Predict probabilities
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display probabilities
        st.header(f"{batting_team} - {round(win * 100)}%")
        st.header(f"{bowling_team} - {round(loss * 100)}%")
    except Exception as e:
        # Handle errors during prediction
        st.error(f"An error occurred during prediction: {e}")
