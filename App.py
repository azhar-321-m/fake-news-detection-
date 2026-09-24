import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="📰 Fake News Detection System",
    page_icon="📰",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------------------------------
# Prediction History
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
# Load Dataset
# -------------------------------
# Load Dataset
# -------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

fake_path = os.path.join(BASE_DIR, "Fake_news", "Fake.csv")
true_path = os.path.join(BASE_DIR, "Fake_news", "True.csv")

fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

# -------------------------------
# Sidebar Navigation
# -------------------------------
page = st.sidebar.radio(
    "📂 Navigation",
    ["🏠 Home", "📊 Dataset Analysis", "ℹ️ About"]
)

st.sidebar.markdown("---")

st.sidebar.title("📌 Project Information")

st.sidebar.success("Machine Learning Project")

st.sidebar.markdown("### 📊 Model Details")
st.sidebar.write("**Algorithm:** Logistic Regression")
st.sidebar.write("**Vectorizer:** TF-IDF")
st.sidebar.write("**Accuracy:** 98.40%")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📚 Dataset")
st.sidebar.write("Total Articles : 44,898")
st.sidebar.write("Fake + True News")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🛠 Technologies")
st.sidebar.write("✔ Python")
st.sidebar.write("✔ Scikit-Learn")
st.sidebar.write("✔ Streamlit")
st.sidebar.write("✔ Pandas")

# ======================================================
# HOME PAGE
# ======================================================

if page == "🏠 Home":

    st.title("📰 Fake News Detection System")

    st.markdown("""
Detect whether a news article is **Fake** or **True**
using a **Machine Learning** model.

Paste any news article below and click **Predict**.
""")

    st.markdown("---")

    news = st.text_area(
        "📝 Enter News Article",
        height=250,
        placeholder="Paste your news article here..."
    )

    if st.button("🔍 Predict", use_container_width=True):

        if news.strip() == "":
            st.warning("⚠ Please enter a news article.")

        else:

            # Vectorize
            news_vector = vectorizer.transform([news])

            # Prediction
            prediction = model.predict(news_vector)

            # Confidence
            probability = model.predict_proba(news_vector)
            confidence = max(probability[0]) * 100

            # Statistics
            words = len(news.split())
            characters = len(news)
            reading_time = max(1, round(words / 200))
            st.session_state.last_result = {
               "prediction": prediction[0],
                "confidence": confidence,
                "words": words,
                "characters": characters,
                "reading_time": reading_time
                }

            # Result
            st.markdown("## 🎯 Prediction Result")

            if prediction[0] == 0:
                st.error("❌ Fake News")
            else:
                st.success("✅ True News")
            # Save Prediction History
            st.session_state.history.append({
              "News": news[:60] + "..." if len(news) > 60 else news,
              "Prediction": "Fake News" if prediction[0] == 0 else "True News",
              "Confidence": f"{confidence:.2f}%"
            })

            st.markdown("---")

            # Dashboard Cards
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Accuracy", "98.40%")

            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")

            with col3:
                st.metric("Reading Time", f"{reading_time} min")

            st.markdown("---")

            # Progress Bar
            st.subheader("📈 Model Confidence")

            st.progress(int(confidence))

            st.write(f"Confidence Score : **{confidence:.2f}%**")

            st.markdown("---")

            # Statistics
            st.subheader("📝 Article Statistics")

            col4, col5 = st.columns(2)

            with col4:
                st.metric("Words", words)

            with col5:
                st.metric("Characters", characters)
            st.markdown("---")
            st.subheader("🕒 Prediction History")

            if len(st.session_state.history) > 0:
             history_df = pd.DataFrame(st.session_state.history)
             st.dataframe(history_df, use_container_width=True)
             csv = history_df.to_csv(index=False).encode("utf-8")

             st.download_button(
             label="📥 Download Prediction History",
             data=csv,
             file_name="prediction_history.csv",
             mime="text/csv"
             )
            if st.button("🗑️ Clear History"):
             st.session_state.history = []
             st.success("Prediction history cleared successfully!")
             
            else:
             st.info("No predictions yet.")

# ======================================================
# DATASET PAGE
# ======================================================

elif page == "📊 Dataset Analysis":

    st.title("📊 Dataset Analysis")

    total_articles = len(fake_df) + len(true_df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Articles", total_articles)

    with col2:
        st.metric("Fake News", len(fake_df))

    with col3:
        st.metric("True News", len(true_df))

    st.markdown("---")

    st.subheader("📋 Dataset Preview")

    st.write("### Fake News Sample")
    st.dataframe(fake_df.head())

    st.write("### True News Sample")
    st.dataframe(true_df.head())
    st.markdown("---")
    st.subheader("📊 Fake vs True News Distribution")

    news_counts = [len(fake_df), len(true_df)]
    labels = ["Fake News", "True News"]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(labels, news_counts)

    ax.set_ylabel("Number of Articles")
    ax.set_title("Fake vs True News Distribution")

    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🥧 Fake vs True News Percentage")

    fig = plt.figure(figsize=(3, 3))
    plt.pie(
    [len(fake_df), len(true_df)],
    labels=["Fake News", "True News"],
    autopct="%1.1f%%",
    startangle=90
   )
    plt.title("Fake vs True News Percentage")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("📰 News by Subject")

# Combine both datasets
    combined_df = pd.concat([fake_df, true_df])

# Count articles by subject
    subject_counts = combined_df["subject"].value_counts()

# Create bar chart
    fig = plt.figure(figsize=(4, 1.5))

    plt.bar(subject_counts.index, subject_counts.values)

    plt.title("News Articles by Subject")
    plt.xlabel("Subject")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=45)

    st.pyplot(fig)
    plt.close(fig)


    st.markdown("---")
    st.subheader("☁️ Fake News Word Cloud")

# Combine all fake news text
    fake_text = " ".join(fake_df["text"].fillna("").astype(str))

# Generate Word Cloud
    wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    max_words=200
    ).generate(fake_text)

# Display
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("☁️ True News Word Cloud")

# Combine all true news text
    true_text = " ".join(true_df["text"].fillna("").astype(str))

# Generate Word Cloud
    wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    max_words=200
    ).generate(true_text)

# Display
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)

# ======================================================
# ABOUT PAGE
# ======================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""
## 📰 Fake News Detection System

This project predicts whether a news article is **Fake** or **True**
using Machine Learning.

### Model
- Logistic Regression
- TF-IDF Vectorizer

### Dataset
- Fake.csv
- True.csv
- Total Articles: 44,898

### Technologies
- Python
- Streamlit
- Pandas
- Scikit-Learn

### Accuracy
98.40%
""")