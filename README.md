📰 Fake News Detection Using Machine Learning


🚀 Real-Time Fake News Classifier with Explainability
A machine learning project to detect fake news articles using NLP and Logistic Regression/SVM. The system features a real-time web interface and explains why a news article is predicted as fake or real.

--------------------------------------------------------------------------

📌 Features

✅ Logistic Regression (or Linear SVM) model trained on real-world news datasets.

✅ NLP preprocessing: text cleaning, TF-IDF vectorization.

✅ 99% accuracy on internal data.

✅ Real-time prediction using a beautiful Streamlit web app.

✅ Explainable AI with eli5 — shows top words influencing the decision.

✅ Proper data validation, train/test split, and model saving/loading.

✅ Ethical considerations discussed: potential misuse, bias, and real-world deployment caution.

-----------------------------------------------------------------------------

📊 Model Training & Evaluation

Text Features: Combined title and body → cleaned → TF-IDF vectorized.

Algorithm: Logistic Regression / LinearSVC

Split: Stratified train/val/test (80/10/10)


Metrics:
    Accuracy: ~99% (internal)
    Precision & Recall: High and balanced
    Confusion Matrix: Evaluated and included

----------------------------------------------------------

🧠 Explainability with eli5

The model doesn't just say “fake” — it tells you why:
        "The model focused on: hillary, conspiracy"

This is implemented using eli5.explain_prediction, and the app shows the top 1–2 keywords influencing the decision.

-----------------------------------------------------------------------------

🖥️ How to Run

🧪 1. Install Dependencies:

        pip install -r requirements.txt
Or manually:
    
        pip install streamlit scikit-learn pandas numpy eli5

🚀 2. Start the Streamlit App

    streamlit run app/streamlit_app.py
    
The app will open in your browser at http://localhost:8501

--------------------------------------------------------------------------------

🔎 Dataset
Source: Combined from Kaggle Fake and Real News datasets

Total Articles: ~12,000+

Balanced across fake/real

Preprocessing: clean text, drop nulls, merge title + body, stratified split

---------------------------------------------------------------------------------

⚖️ Ethical Considerations

Fake news detection is not a solved problem.

This project is an educational showcase, not a production-ready fact-checking engine. We recognize:

✋ Bias in training data can lead to unfair labeling
    
🔄 The model may mislabel satire or opinion as fake
    
🧪 Always combine ML systems with human oversight

---------------------------------------------------------------------------------

💡 What You Learn from This Project

✅ How to build an NLP pipeline

✅ Train and evaluate classification models

✅ Save and load models using joblib

✅ Build an interactive UI with Streamlit

✅ Explain predictions with eli5

✅ Communicate ethical issues in AI responsibly

--------------------------------------------------------------------------------

📌 Future Enhancements

🧠 Use BERT or LLMs for better context understanding

🌐 Deploy to Streamlit Cloud / Hugging Face Spaces

📱 Turn into a mobile-responsive app

🔍 Integrate fact-checking APIs for validation

📦 Bundle with Docker for reproducibility

---------------------------------------------------------------------------------

🔗 Try the live app here: your-app.streamlit.app
