# 📰 Fake News Detection System

A Machine Learning-based web application that predicts whether a news article is **Fake** or **True** using TF-IDF Vectorization and Logistic Regression.

## 📌 Project Overview

The Fake News Detection System classifies news articles into two categories:

- ✅ True News
- ❌ Fake News

The project uses **TF-IDF Vectorization** to convert text into numerical features and **Logistic Regression** for classification.

The trained model is integrated with a **Streamlit web application** for real-time prediction.

## 🚀 Features

- Fake and True news classification
- TF-IDF text vectorization
- Logistic Regression model
- Prediction probability
- User-friendly Streamlit interface
- Real-time prediction
- Pre-trained model and vectorizer

## 🤖 Machine Learning Model

### Algorithm
**Logistic Regression**

### Feature Extraction
**TF-IDF Vectorizer**

### Problem Type
**Binary Classification**

## 📊 Model Performance

**Accuracy: 98.40%**

## 📂 Dataset

The dataset used for this project was obtained from **Kaggle**.

The dataset contains labeled news articles categorized as Fake and True.

- Total Articles: **44,898**
- Categories: **Fake, True**

> **Note:** The original dataset files are not included in this repository because the dataset was obtained from Kaggle.

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **TF-IDF**
- **Logistic Regression**

## 📁 Project Structure

```text
fake-news-detection/
│
├── App.py
├── Train.py
├── model.pkl
├── vectorizer.pkl
