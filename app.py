import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AlgoAditya Estate Advisor", page_icon="🏡", layout="wide"
)


# --- LOAD DATA & MODEL ---
@st.cache_data
def load_resources():
    # Load the CSV just to get the location names
    data = pd.read_csv("cleaned_data.csv")

    # Load the Random Forest Model
    # MAKE SURE 'RandomForestModel.pkl' exists in your repo!
    pipe = pickle.load(open("RandomForestModel.pkl", "rb"))  # <--- NEW FIXED FILE
    return data, pipe


try:
    df, model = load_resources()
except FileNotFoundError:
    st.error(
        "Error: Could not find 'RandomForestModel.pkl'. Please run 'python random_forest.py' first."
    )
    st.stop()

# --- SIDEBAR (User Inputs) ---
st.sidebar.header("🔍 Find Your Home")
st.sidebar.write("Enter property details below:")

# 1. Location Selection
locations = sorted(df["location"].unique())
selected_location = st.sidebar.selectbox("Select Location", locations)

# 2. BHK & Bathrooms
col1, col2 = st.sidebar.columns(2)
with col1:
    bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2)
with col2:
    bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

# 3. Area (Square Feet)
sqft = st.sidebar.number_input(
    "Total Sqft Area", min_value=300, max_value=20000, value=1200
)

# --- MAIN PAGE DESIGN ---
st.title("🏡 Bangalore Real Estate Advisor")
st.markdown(f"### Predicting prices for properties in **{selected_location}**")

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict Price", type="primary"):
    try:
        # Create a dataframe with inputs
        # The column names MUST match X from random_forest.py
        input_data = pd.DataFrame(
            [[selected_location, sqft, bath, bhk]],
            columns=["location", "total_sqft", "bath", "bhk"],
        )

        # Predict
        prediction = model.predict(input_data)[0]

        # Display Result
        st.success("✅ Estimated Market Value")
        st.metric(label="Price (Lakhs)", value=f"₹ {prediction:.2f} L")

        # --- EXTRA FEATURE: Price per Sqft Analysis ---
        price_per_sqft = (prediction * 100000) / sqft
        st.info(f"📊 Market Rate: **₹ {price_per_sqft:.0f} / sqft**")

        if price_per_sqft < 5000:
            st.write("🔥 **Insight:** This area is relatively affordable.")
        elif price_per_sqft > 15000:
            st.write("💎 **Insight:** This is a premium/luxury area.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.warning(
            "Debugging Hint: Check if input columns match model training columns."
        )

# --- FOOTER ---
st.markdown("---")
st.caption("Built by **AlgoAditya** | powered by Scikit-Learn & Streamlit")
