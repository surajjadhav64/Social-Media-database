import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import  CountVectorizer

with open('count_vectorizer.pkl','rb')as vectorizer_file:
    count_vectorizer = pickle.load(vectorizer_file)

with open('nb_classifier.pkl','rb')as classifier_file:
    nb_classifier = pickle.load(classifier_file)

def process_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

sentiment_mapping = {
        "Negative" : "Negative 😔",
        "Positive" : "Positive 😊",
        "Neutral" : "Neutral 🙄",
        "Irrelevant" : "Irrelevant 🤷‍♂️"
    }

def main():
    col1 , col2 , col3 ,col4 = st.columns([1,1,3,1])
    with col3:
        st.image("./image/pngwing.com (1).png" , width=100)
    st.title("Twitter Sentiment Classifier")
    st.write("Enter twitter tweet below :")
    input_text = st.text_area("Input Text :","")
    if st.button("Predict"):
        cleaned_text = process_text(input_text)
        vectorizer_text = count_vectorizer.transform([cleaned_text])
        sentiment_prediction = nb_classifier.predict(vectorizer_text)[0]
        
        predicted_sentiment =  sentiment_mapping.get(sentiment_prediction , "Unknown Sentiment")

        st.write("Predicted Sentimen :")
        st.title(predicted_sentiment)


if __name__  == "__main__":
    main()