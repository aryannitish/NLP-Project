import streamlit as st
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Preprocessing function for input text
def preprocess_input_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)  # Remove HTML line breaks
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Train model function
def train_model(data):
    data["review"] = data["review"].apply(preprocess_input_text)
    X = data["review"]
    y = data["sentiment"].apply(lambda x: 1 if x == "positive" else 0)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vectorized, y_train)

    # Evaluate the model
    accuracy = accuracy_score(y_test, model.predict(X_test_vectorized))

    # Save the vectorizer and model
    with open("tfidf_vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    with open("logistic_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    return accuracy

# Predict sentiment
def predict_sentiment(text, model, vectorizer):
    cleaned_text = preprocess_input_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return sentiment

# Streamlit app
st.title("Sentiment Analysis App")

# File upload for training
data_file = st.file_uploader("Upload a CSV file for training (with 'review' and 'sentiment' columns):", type=["csv"])

if data_file:
    data = pd.read_csv(data_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    if st.button("Train Model"):
        accuracy = train_model(data)
        st.write(f"Model trained successfully with an accuracy of {accuracy:.2f}!")

st.write("Enter a movie review to predict its sentiment:")

# Input review text
user_input = st.text_area("Review", "")

if st.button("Predict Sentiment"):
    if user_input:
        try:
            with open("tfidf_vectorizer.pkl", "rb") as vec_file:
                tfidf_vectorizer = pickle.load(vec_file)
            with open("logistic_model.pkl", "rb") as model_file:
                logistic_model = pickle.load(model_file)

            # Predict sentiment
            sentiment = predict_sentiment(user_input, logistic_model, tfidf_vectorizer)

            # Display result
            st.write(f"Predicted Sentiment: **{sentiment}**")
        except FileNotFoundError as e:
            st.error("Model not found. Please train the model first by uploading a dataset.")
    else:
        st.write("Please enter a review.")
