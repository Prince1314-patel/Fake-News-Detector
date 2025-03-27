import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from joblib import load
import pandas as pd

# Cache the model loading
@st.cache_resource
def load_model():
    model_path = "news_classifier_model.pkl"
    return load(model_path)

# Load the model (this will be cached)
model = load_model()

# Cache the preprocessing function
@st.cache_data
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Streamlit app
st.title("Fake News Detector")
st.write("Enter a news article below, and the model will classify it as Fake or Real.")

# Input text
user_input = st.text_area("Enter news text here:")

if st.button("Classify"):
    if user_input.strip():
        # Preprocess the input using the cached function
        processed_text = preprocess_text(user_input)
        # Convert to a DataFrame with the column name 'text'
        input_df = pd.DataFrame({'text': [processed_text]})
        try:
            # Make prediction using the DataFrame
            prediction = model.predict(input_df)[0]
            # Display the result
            if prediction == 1:
                st.success("The news is classified as **Real**.")
            else:
                st.error("The news is classified as **Fake**.")
        except ValueError as e:
            st.error(f"Model error: {e}")
    else:
        st.warning("Please enter some text to classify.")