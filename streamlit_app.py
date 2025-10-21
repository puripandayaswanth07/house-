# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- MODEL LOADING AND SETUP ---

# This function is cached, so it only runs once
@st.cache_resource
def load_model_and_encoders():
    """
    Loads the trained model and encoders. If they don't exist,
    it creates and trains a new sample model.
    """
    from data_preparation import get_chennai_localities
    
    model_path = 'model/chennai_house_model.joblib'
    location_encoder_path = 'model/location_encoder.joblib'
    
    # Create directories if they don't exist
    os.makedirs('model', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # If model doesn't exist, create a sample one
    if not os.path.exists(model_path):
        st.warning("Model not found. Creating and training a new sample model... This may take a moment.")
        model, location_encoder, locations = create_sample_model()
    else:
        model = joblib.load(model_path)
        location_encoder = joblib.load(location_encoder_path)
        
        # Ensure we have the most comprehensive list of locations
        all_locations = sorted(get_chennai_localities())
    
    return model, location_encoder, all_locations

def create_sample_model():
    """Create and train a simple model for demonstration."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from data_preparation import get_chennai_localities
    
    chennai_localities = get_chennai_localities()
    
    np.random.seed(42)
    n_samples = max(2000, len(chennai_localities) * 5)
    
    locations = list(chennai_localities) * 5
    locations += np.random.choice(chennai_localities, n_samples - len(locations)).tolist()
    
    data = {
        'total_sqft': np.random.randint(400, 6000, n_samples),
        'bedrooms': np.random.randint(1, 7, n_samples),
        'bathrooms': np.random.randint(1, 5, n_samples),
        'location': locations[:n_samples]
    }
    
    base_price_per_sqft = np.random.normal(5500, 1500, n_samples)
    location_factors = {loc: np.random.uniform(0.6, 1.8) for loc in chennai_localities}
    
    data['price_lakhs'] = (
        (data['total_sqft'] * base_price_per_sqft * [location_factors[loc] for loc in data['location']]) / 100000
    )
    
    df = pd.DataFrame(data)
    df.to_csv('data/processed/chennai_house_data.csv', index=False)
    
    X = df[['total_sqft', 'bedrooms', 'bathrooms', 'location']]
    y = df['price_lakhs']
    
    le = LabelEncoder()
    le.fit(chennai_localities)
    
    X['location_encoded'] = le.transform(X['location'])
    X = X[['total_sqft', 'bedrooms', 'bathrooms', 'location_encoded']]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, 'model/chennai_house_model.joblib')
    joblib.dump(le, 'model/location_encoder.joblib')
    
    pd.DataFrame({'location': chennai_localities}).to_csv('model/available_locations.csv', index=False)
    
    return model, le, chennai_localities

# Load the resources
try:
    model, location_encoder, available_locations = load_model_and_encoders()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --- STREAMLIT UI ---

st.set_page_config(page_title="Chennai House Price Predictor", layout="wide")

st.title("🏡 Chennai House Price Predictor")
st.markdown("Enter the details of the property to get an estimated price.")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Features")
    total_sqft = st.number_input(
        "Total Square Feet", 
        min_value=300, 
        max_value=10000, 
        value=1200, 
        step=50,
        help="Enter the total area of the house in square feet."
    )
    bedrooms = st.selectbox(
        "Number of Bedrooms (BHK)", 
        options=[1, 2, 3, 4, 5, 6], 
        index=1,
        help="Select the number of bedrooms."
    )
    bathrooms = st.selectbox(
        "Number of Bathrooms", 
        options=[1, 2, 3, 4, 5], 
        index=1,
        help="Select the number of bathrooms."
    )

with col2:
    st.subheader("Location")
    location = st.selectbox(
        "Select a Location", 
        options=available_locations, 
        index=available_locations.index("Adyar") if "Adyar" in available_locations else 0,
        help="Choose the locality in Chennai."
    )

# Prediction button and result display
st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):
    if location in location_encoder.classes_:
        # Encode the location
        location_encoded = location_encoder.transform([location])[0]
        
        # Create input array for the model
        input_data = np.array([[total_sqft, bedrooms, bathrooms, location_encoded]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display the result
        st.success(f"**Predicted Price:**")
        st.metric(label="Estimated Value", value=f"₹ {prediction:,.2f} Lakhs")

        with st.expander("See Prediction Details"):
            st.json({
                "Area": f"{total_sqft} sq.ft",
                "Bedrooms": f"{bedrooms} BHK",
                "Bathrooms": bathrooms,
                "Location": location
            })
    else:
        st.error(f"Location '{location}' is not recognized. Please choose from the available options.")
