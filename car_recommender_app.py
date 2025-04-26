import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import difflib as df

# Load car dataset
car_data = pd.read_csv('car data.csv')

# Ensure column names are clean
car_data.columns = car_data.columns.str.strip().str.lower().str.replace(' ', '_')

# Handle missing values
car_data.fillna('', inplace=True)

# Combine numeric data for collaborative filtering
numeric_data = car_data[['selling_price', 'kms_driven']].fillna(0)

# Content features for content-based filtering
car_data['combined_features'] = (
    car_data['fuel_type'].astype(str) + ' ' +
    car_data['seller_type'].astype(str) + ' ' +
    car_data['transmission'].astype(str) + ' ' +
    car_data['owner'].astype(str)
)

# Function to find the closest match
def find_closest_match(user_input):
    car_names = car_data['car_name'].tolist()
    closest_matches = df.get_close_matches(user_input, car_names, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None

# Collaborative Filtering
def collaborative_filtering(car_name):
    similarity_matrix = cosine_similarity(numeric_data)
    car_index = car_data[car_data['car_name'] == car_name].index[0]
    similar_cars = list(enumerate(similarity_matrix[car_index]))
    sorted_similar_cars = sorted(similar_cars, key=lambda x: x[1], reverse=True)
    return sorted_similar_cars[1:11]  # Exclude itself

# Content-Based Filtering
def content_based_filtering(car_name):
    cv = CountVectorizer()
    features_matrix = cv.fit_transform(car_data['combined_features'])
    similarity_scores = cosine_similarity(features_matrix, features_matrix)
    car_index = car_data[car_data['car_name'] == car_name].index[0]
    similar_cars = list(enumerate(similarity_scores[car_index]))
    sorted_similar_cars = sorted(similar_cars, key=lambda x: x[1], reverse=True)
    return sorted_similar_cars[1:11]  # Exclude itself

# Streamlit UI
st.title("🚗 Car Recommender System")

# User input
car_name = st.text_input("Enter the name of a car:")

# Filtering method
filtering_method = st.selectbox("Select Recommendation Method:", ["Collaborative Filtering", "Content-Based Filtering"])

# Find best match
closest_match = find_closest_match(car_name)

if closest_match:
    st.success(f"Closest match found: {closest_match}")

    # Display the car details for the closest match
    car_details = car_data[car_data['car_name'] == closest_match].iloc[0]
    st.subheader(f"Details for '{closest_match}':")
    st.write(f"**Selling Price:** ₹{car_details['selling_price']}")
    st.write(f"**Kilometers Driven:** {car_details['kms_driven']} km")
    st.write(f"**Fuel Type:** {car_details['fuel_type']}")
    st.write(f"**Transmission:** {car_details['transmission']}")
    st.write(f"**Owner:** {car_details['owner']}")
    
    # Recommendations based on the selected method
    if filtering_method == "Collaborative Filtering":
        recommendations = collaborative_filtering(closest_match)
        st.subheader(f"Top 10 cars similar to '{closest_match}' (Collaborative Filtering):")
        for car in recommendations:
            st.write(f"- {car_data.iloc[car[0]]['car_name']} (Similarity Score: {car[1]:.4f})")

    elif filtering_method == "Content-Based Filtering":
        recommendations = content_based_filtering(closest_match)
        st.subheader(f"Top 10 cars similar to '{closest_match}' (Content-Based Filtering):")
        for car in recommendations:
            st.write(f"- {car_data.iloc[car[0]]['car_name']} (Similarity Score: {car[1]:.4f})")
else:
    if car_name:
        st.warning(f"No close match found for: '{car_name}'")
