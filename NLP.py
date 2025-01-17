import streamlit as st
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Preprocessing function for input text
def preprocess_input_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)  # To remove HTML line breaks
    text = re.sub(r"[^a-zA-Z]", " ", text)  # To remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Train model function
def train_models(data):
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
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train_vectorized, y_train)

    # Train the Decision Tree model
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train_vectorized, y_train)

    # Evaluate both models
    logistic_accuracy = accuracy_score(y_test, logistic_model.predict(X_test_vectorized))
    decision_tree_accuracy = accuracy_score(y_test, decision_tree_model.predict(X_test_vectorized))

    # Save the vectorizer and models
    with open("tfidf_vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    with open("logistic_model.pkl", "wb") as logistic_file:
        pickle.dump(logistic_model, logistic_file)

    with open("decision_tree_model.pkl", "wb") as tree_file:
        pickle.dump(decision_tree_model, tree_file)

    return logistic_accuracy, decision_tree_accuracy

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
    logistic_accuracy, decision_tree_accuracy = train_models(data)
    st.write(f"Logistic Regression Model trained successfully with an accuracy of {logistic_accuracy:.2f}")
    st.write(f"Decision Tree Model trained successfully with an accuracy of {decision_tree_accuracy:.2f}")            

st.write("Enter a movie review to predict its sentiment:")

# Input review text
user_input = st.text_area("Review", "")

if st.button("Predict Sentiment"):
    if user_input:
        try:
            # Load vectorizer and models
            with open("tfidf_vectorizer.pkl", "rb") as vec_file:
                tfidf_vectorizer = pickle.load(vec_file)
            with open("logistic_model.pkl", "rb") as logistic_file:
                logistic_model = pickle.load(logistic_file)
            with open("decision_tree_model.pkl", "rb") as tree_file:
                decision_tree_model = pickle.load(tree_file)

            # Predict sentiment using both models
            logistic_sentiment = predict_sentiment(user_input, logistic_model, tfidf_vectorizer)
            decision_tree_sentiment = predict_sentiment(user_input, decision_tree_model, tfidf_vectorizer)

            # Display results
            st.write(f"Logistic Regression Predicted Sentiment: **{logistic_sentiment}**")
            st.write(f"Decision Tree Predicted Sentiment: **{decision_tree_sentiment}**")
        except FileNotFoundError as e:
            st.error("Models not found. Please train the models first by uploading a dataset.")
    else:
        st.write("Please enter a review.")
