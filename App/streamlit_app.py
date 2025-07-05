import time
import streamlit as st
import joblib
import os
import eli5
from eli5.sklearn import explain_prediction


#Loading and vectorization of the model
model=joblib.load('Models/logistic_model.pkl')
tfidf=joblib.load('Models/tfidf_vectorizer.pkl')

# Now we would use a simple text-processing function which does ->
# 1.Converts text to lowercase
# 2. Removes new line characters
# 3. Removes all characters except -letters, number,punctuation marks and spaces
# 4. Collapses multiple spaces into one

import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,;!?()\'" ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



#Now streamlit UI
# 1. Sets the page configuration(title,layout)
# 2. Displays a page title and short introduction 
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: #3366cc;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a news article or headline below. This app uses a machine learning model to detect whether it's real or fake.</p>", unsafe_allow_html=True)

# Sidebar for info
with st.sidebar:
    st.title("ℹ️ About")
    st.info("""
    This app uses a **Logistic Regression model** trained on news data using **TF-IDF** features. 
    
    Built with ❤️ using Streamlit and scikit-learn.
    """)
    st.markdown("[GitHub Repo](https://github.com/gauriRajpal/Fake_news_detector)", unsafe_allow_html=True)

#Text input->
    # 1. Creates a text area box where users can write the news

input_text = st.text_area("Paste News Text Here", height=200)


# Button to trigger prediciton-> Adds a "Detect" button 
if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    #Checks if the input is empty or whitespace and if there is then shows the warning message    
    else:
        with st.spinner("Analysing..."):
            time.sleep(1)
        #Cleans the text
        clean_input = clean_text(input_text)

        #Transforms it to TF-IDF Features using the tfidf vectorizer
        vectorized = tfidf.transform([clean_input])

        #Feeds it to the model to predict the class(0==real,1==fake)
        prediction = model.predict(vectorized)[0]

        #Get the decision score to tell how confident the model is
        proba = model.decision_function(vectorized)[0]

        confidence=min(abs(proba),1)        #Bound between 0 and 1

        #Coverts the prediction in readable form
        if prediction == 1:
                st.error("🔴 **Fake News Detected**", icon="🚨")
        else:
                st.success("🟢 **Real News Detected**", icon="✅")

        # Progress bar as confidence indicator
        st.markdown("### 🔎 Confidence Level")
        st.progress(confidence)

        # Numerical score
        st.caption(f"Model certainty score: `{abs(proba):.2f}` (closer to 1 = more confident)")

        # Optional: Helpful message
        if confidence < 0.3:
            st.info("⚠️ This prediction is uncertain. Consider fact-checking this article yourself.")
        

        # Explanation using eli5
        st.markdown("### 🧠 Top word(s) influencing this prediction:")
        explanation = eli5.explain_prediction(
        model,
        clean_input,
        vec=tfidf,
        top=2
        )

        # Access weights correctly based on prediction
    if prediction == 1:
        weights = explanation.targets[0].feature_weights.pos  # For label 1 (fake)
    else:
        weights = explanation.targets[0].feature_weights.neg  # For label 0 (real)

    # Extract top 2 word features
    important_words = [w.feature for w in weights if w.feature.strip() != '' and w.feature != '<BIAS>'][:2]

    if important_words:
        st.write(f"🔍 The model focused on: **{', '.join(important_words)}**")
    else:
        st.info("🤖 The model made this decision based on general patterns, not specific keywords.")

# Footer
st.markdown("---")
