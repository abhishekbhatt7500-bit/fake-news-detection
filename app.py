
import streamlit as st
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="AI News Analyzer",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: white;
}
.main-title {
    text-align: center;
    color: #38bdf8;
    font-size: 42px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">📰 AI NEWS ANALYZER</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Fake News Detection System | Hindi + English</div>',
    unsafe_allow_html=True
)

st.divider()

# Sidebar
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "About Project"]
)

model_path = "fake_news_model.pkl"

if page == "Home":

    st.header("🔍 Analyze Your News")

    if os.path.exists(model_path):

        try:
            with open(model_path, "rb") as file:
                model = pickle.load(file)

            news = st.text_area(
                "Enter your news here:",
                height=180,
                placeholder="Paste your news here..."
            )

            if st.button("🚀 Analyze News", use_container_width=True):

                if news.strip():

                    prediction = model.predict([news])[0]

                    st.subheader("📊 Analysis Result")

                    if str(prediction).upper() == "REAL":
                        st.success("✅ REAL NEWS")
                    else:
                        st.error("⚠️ FAKE NEWS")

                    st.info(
                        "This is an AI model prediction, "
                        "not verified fact-checking."
                    )

                else:
                    st.warning("Please enter some news first.")

        except Exception as e:
            st.error(f"Model loading or prediction error: {e}")

    else:
        st.error("Model file not found: fake_news_model.pkl")

elif page == "About Project":

    st.header("📚 About This Project")

    st.write("""
    This project uses Machine Learning to classify news.

    **Features:**
    - Fake News Detection
    - Real News Detection
    - News Text Analysis
    - Simple Dashboard

    **Technology Used:**
    - Python
    - Machine Learning
    - Streamlit
    - Scikit-learn

    **Note:** Predictions depend on the training data
    and model quality. They are not guaranteed to be correct.
    """)

st.divider()

st.caption("AI News Analyzer | Academic AIML Project")
