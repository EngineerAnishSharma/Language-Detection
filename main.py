import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('Language Detection.csv')

# Preprocess the data
x = np.array(df['Text'])
y = np.array(df['Language'])

cv = CountVectorizer(lowercase=True)
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB(alpha=0.05)
model.fit(X_train, y_train)

# Streamlit app
st.title("Language Detection Model")
st.write("This app predicts the language of the input text.")

# User input
user_input = st.text_area("Enter text for language detection:")

if st.button("Detect Language"):
    if user_input:
        # Transform user input and predict language
        enc_input = cv.transform([user_input])
        output = model.predict(enc_input)
        st.success(f'The detected language is: {output[0]}')
    else:
        st.error("Please enter some text.")

# Optional: Display a Word Cloud for the English language
if st.checkbox("Show Word Cloud for English Language"):
    text = " ".join(df[df['Language'] == 'English']['Text'])
    text = text.lower()
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=set(stopwords.words('english'))).generate(text)
    st.image(wordcloud.to_array(), caption='Word Cloud for English Language', use_column_width=True)

