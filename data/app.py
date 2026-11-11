
import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰"
)

st.title("📰 Fake News Detection System")
st.subheader("Hindi + English News Checker")

model_path = "fake_news_model.pkl"

if os.path.exists(model_path):

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    news = st.text_area("Enter your news:")

    if st.button("Check News"):
        if news.strip():
            prediction = model.predict([news])[0]

            if prediction == "REAL":
                st.success("Prediction: REAL NEWS")
            else:
                st.error("Prediction: FAKE NEWS")

        else:
            st.warning("Please enter news first.")

    st.caption("Experimental prediction, not verified fact-checking.")

else:
    st.error("Model file not found. Please train the model first.")
