import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="AlgoAditya Estate Advisor",
    page_icon="🏡",
    layout="wide"
)

# --- LOAD DATA & MODEL ---
@st.cache_data # Caches the data so it doesn't reload every time you click a button
def load_resources():
    data = pd.read_csv("cleaned_data.csv")
    # REPLACE 'BestModel.pkl' with your actual filename (e.g., RidgeModel.pkl)
    pipe = pickle.load(open("RidgeModel.pkl", "rb")) 
    return data, pipe

df, model = load_resources()

# --- SIDEBAR (User Inputs) ---
st.sidebar.header("🔍 Find Your Home")
st.sidebar.write("Enter property details below:")

# 1. Location Selection
locations = sorted(df['location'].unique())
location = st.sidebar.selectbox("Select Location", locations)

# 2. BHK & Bathrooms
col1, col2 = st.sidebar.columns(2)
with col1:
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
with col2:
    bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

# 3. Area (Square Feet)
sqft = st.sidebar.number_input("Total Sqft Area", min_value=300, max_value=20000, value=1200)

# --- MAIN PAGE DESIGN ---
st.title("🏡 Bangalore Real Estate Advisor")
st.markdown(f"### Predicting prices for properties in **{location}**")

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict Price", type="primary"):
    try:
        # Create a dataframe matching the training format EXACTLY
        input_data = pd.DataFrame([[location, sqft, bath, bhk]], 
                                  columns=['location', 'total_sqft', 'bath', 'bhk'])
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        # Display Result with a nice metric card
        st.success("✅ Estimated Market Value")
        st.metric(label="Price (Lakhs)", value=f"₹ {prediction:.2f} L")
        
        # --- EXTRA FEATURE: Price per Sqft Analysis (For Rubric Depth) ---
        price_per_sqft = (prediction * 100000) / sqft
        st.info(f"📊 Market Rate: **₹ {price_per_sqft:.0f} / sqft**")
        
        if price_per_sqft < 5000:
            st.write("🔥 **Insight:** This area is relatively affordable.")
        elif price_per_sqft > 15000:
            st.write("💎 **Insight:** This is a premium/luxury area.")
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("Make sure your model file is named correctly in the code!")

# --- FOOTER ---
st.markdown("---")
st.caption("Built by **AlgoAditya** | powered by Scikit-Learn & Streamlit")