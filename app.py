import streamlit as st
import pandas as pd
import pickle
from agent import get_real_estate_agent

st.set_page_config(page_title="AlgoAditya Estate Advisor",page_icon="🏡",layout="wide")

@st.cache_data
def load_resources():
    data=pd.read_csv("cleaned_data.csv")
    pipe=pickle.load(open("RandomForestModel.pkl","rb"))
    return data, pipe

try:
    df, model=load_resources()
except FileNotFoundError:
    st.error("Error: Could not find 'RandomForestModel.pkl'. Please run 'python random_forest.py' first.")
    st.stop()

#sidebar
st.sidebar.header("🔍 Find Your Home")
st.sidebar.write("Enter property details below:")

#location selection
locations=sorted(df["location"].unique())
selected_location=st.sidebar.selectbox("Select Location",locations)

#BHK & Bathrooms
col1,col2=st.sidebar.columns(2)
with col1:
    bhk=st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2)
with col2:
    bath=st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

#area sqft
sqft=st.sidebar.number_input("Total Sqft Area", min_value=300, max_value=20000, value=1200)

#MAIN PAGE
st.title("🏡 Bangalore Real Estate Advisor")
st.markdown(f"### Predicting prices for properties in **{selected_location}**")

# Initialize the AI Agent only once to save loading time
if "agent" not in st.session_state:
    try:
        st.session_state.agent = get_real_estate_agent()
        st.session_state.agent_error = None
    except Exception as e:
        st.session_state.agent = None
        st.session_state.agent_error = str(e)

# Initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I am your AI Property Advisor. Once you generate a price prediction above, ask me any questions about the location, market trends, or whether it's a good investment!",
        }
    ]

if st.sidebar.button("Predict Price", type="primary"):
    try:
        input_data=pd.DataFrame([[selected_location,sqft,bath,bhk]],columns=["location","total_sqft","bath","bhk"],)
        #predict
        prediction=model.predict(input_data)[0]
        #display result
        st.success("✅ Estimated Market Value")
        st.metric(label="Price (Lakhs)",value=f"₹ {prediction:.2f} L")

        price_per_sqft=(prediction*100000)/sqft
        st.info(f"📊 Market Rate: **₹ {price_per_sqft:.0f} / sqft**")

        if price_per_sqft<5000:
            st.write("🔥 **Insight:** This area is relatively affordable.")
        elif price_per_sqft>15000:
            st.write("💎 **Insight:** This is a premium/luxury area.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("Debugging Hint: Check if input columns match model training columns.")

st.markdown("---")

st.subheader("🤖 AI Property Advisor (RAG Agent)")

# 1. Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.get("agent_error"):
    st.warning(
        "AI advisor could not be initialized. "
        f"Reason: {st.session_state['agent_error']}"
    )
else:
    # 2. Create the chat input box at the bottom
    if prompt := st.chat_input("Ask about the location, amenities, or investment potential..."):
        # Show the user's message on screen
        with st.chat_message("user"):
            st.markdown(prompt)

        # Save the user's message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 3. Generate and show the AI's response
        with st.chat_message("assistant"):
            with st.spinner("Searching market reports..."):
                # Pass the user's question to the RAG Agent
                response = st.session_state.agent.invoke({"input": prompt})
                answer = response["answer"]

                st.markdown(answer)

                # Save the AI's response to history
                st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption("Built by **AlgoAditya** | powered by Scikit-Learn & Streamlit")
