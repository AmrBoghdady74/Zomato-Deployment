
# import libraries
import streamlit as st
import pandas as pd
import joblib
import sklearn
import category_encoders

# Define paths
MODEL_PATH = 'last_model.pkl'
INPUTS_PATH = 'last_inputs.pkl'

# Read model and inputs
model = joblib.load(MODEL_PATH)
inputs = joblib.load(INPUTS_PATH)

# Define function to make prediction
def make_prediction(online_order, book_table, location, approx_cost, listed_in_type, listed_in_city, rest_type_counts, cuisines_counts):
    df_pred = pd.DataFrame(columns=inputs)
    df_pred.at[0, "online_order"] = online_order
    df_pred.at[0, "book_table"] = book_table
    df_pred.at[0, "location"] = location
    df_pred.at[0, "approx_cost(for two people)"] = approx_cost
    df_pred.at[0, "listed_in(type)"] = listed_in_type
    df_pred.at[0, "listed_in(city)"] = listed_in_city
    df_pred.at[0, "rest_type_counts"] = rest_type_counts
    df_pred.at[0, "cuisines_counts"] = cuisines_counts
    proba = model.predict_proba(df_pred)[0]
    result = model.predict(df_pred)[0]  # Get the predicted class
    # Estimate confidence as the probability of the predicted class
    confidence = round(proba[result] * 100, 2)
    return result, confidence

# Define main function to create the Streamlit app
def main():
    st.title('Bangalore Restaurants Success Predictor')
    st.write('Welcome to Bangalore Restaurants Success Predictor. This app helps you predict the success of your restaurant based on various factors.')

    # Add an image
    #image = Image.open('restaurant.jpg')
    #st.image(image, caption='Bangalore Restaurants')

    # Add inputs
    st.sidebar.header('Enter Restaurant Details')
    online_order = st.sidebar.radio('Restaurant has online ordering:', ['Yes', 'No'])
    book_table = st.sidebar.radio('Restaurant has ability to book a table:', ['Yes', 'No'])
    location = st.sidebar.selectbox('Restaurant location:', ['Banashankari', 'Basavanagudi', 'other', 'Jayanagar', 'JP Nagar',
                                                              'Bannerghatta Road', 'BTM', 'Electronic City', 'Shanti Nagar',
                                                              'Koramangala 5th Block', 'Richmond Road', 'HSR',
                                                              'Koramangala 7th Block', 'Bellandur', 'Sarjapur Road',
                                                              'Marathahalli', 'Whitefield', 'Old Airport Road', 'Indiranagar',
                                                              'Koramangala 1st Block', 'Frazer Town', 'MG Road', 'Brigade Road',
                                                              'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road',
                                                              'Shivajinagar', 'St. Marks Road', 'Cunningham Road',
                                                              'Commercial Street', 'Vasanth Nagar', 'Domlur',
                                                              'Koramangala 8th Block', 'Ejipura', 'Jeevan Bhima Nagar',
                                                              'Kammanahalli', 'Koramangala 6th Block', 'Brookefield',
                                                              'Koramangala 4th Block', 'Banaswadi', 'Kalyan Nagar',
                                                              'Malleshwaram', 'Rajajinagar', 'New BEL Road'])
    approx_cost = st.sidebar.slider('Approximation cost for two people:', min_value=10, max_value=10000, value=800, step=200)
    listed_in_type = st.sidebar.selectbox('Restaurant listed in type:', ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
                                                                         'Drinks & nightlife', 'Pubs and bars'])
    listed_in_city = st.sidebar.selectbox('Restaurant listed in city:', ['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
                                                                         'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
                                                                         'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
                                                                         'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
                                                                         'Koramangala 4th Block', 'Koramangala 5th Block',
                                                                         'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
                                                                         'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
                                                                         'Old Airport Road', 'Rajajinagar', 'Residency Road',
                                                                         'Sarjapur Road', 'Whitefield'])
    rest_type_counts = st.sidebar.selectbox('Restaurant type counts:', [1, 2])
    cuisines_counts = st.sidebar.selectbox('Restaurant cuisine counts:', [1, 2, 3, 4, 5, 6, 7, 8])

    if st.button('Predict'):
        result, confidence = make_prediction(online_order, book_table, location, approx_cost, listed_in_type, listed_in_city, rest_type_counts, cuisines_counts)
        list_types = ['Your restaurant may fail', 'Your restaurant will succeed']
        st.write(f"**Prediction:** {list_types[result]} :crystal_ball:")
        st.write(f"**Confidence:** {confidence}%")

# Call the main function to run the app
if __name__ == "__main__":
    main()
