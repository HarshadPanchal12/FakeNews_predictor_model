📰 Fake News Predictor Model
📘 Overview

The Fake News Predictor Model is a Machine Learning project that classifies news articles as Real or Fake using text analysis and metadata.
It’s a part of my extended research work on “Modeling Fake News Spread using ODEs and Machine Learning.”
The project aims to combine mathematical modeling (using ODEs) with AI-based fake news detection for better understanding and controlling misinformation spread.

🎯 Objectives

Detect and classify fake vs real news accurately.

Analyze fake news spread using Ordinary Differential Equations (ODEs).

Integrate simulation + ML prediction results for research publication.

Provide an easy-to-use interface for testing any news text.

🧠 Project Architecture

Data Preprocessing

Merged politifact_fake.csv and politifact_real.csv.

Cleaned, normalized, and labeled the dataset.

Feature Engineering

Extracted text features using TF-IDF Vectorization.

Added optional metadata (engagement, author info, etc.).

Model Training

Algorithms used:

Logistic Regression

Random Forest

Naïve Bayes

(Optionally) SVM / LSTM

Model Evaluation

Accuracy, Precision, Recall, F1-score

Confusion Matrix and ROC Curve visualizations

Prediction App

User inputs any news text and gets prediction (Real or Fake).

Built using Streamlit or Flask for web interface.

🧮 Mathematical Model (ODE Section)

Alongside the ML model, a system of Ordinary Differential Equations (ODEs) simulates how fake news spreads through social networks.
The ODE model studies:

User interaction rate

Retweet probability

Fake vs real content exposure
This helps correlate news virality with predicted authenticity.
