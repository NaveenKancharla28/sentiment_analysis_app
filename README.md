# 🧠 Sentiment Analysis App using Sentiment140

This project is a simple yet powerful sentiment analysis tool built using the **Sentiment140 dataset**. It uses machine learning to classify text data (tweets/reviews) into **positive** or **negative** sentiment.

---

## 🔍 Project Overview

- ✅ Cleaned and preprocessed real-world Twitter data
- ✅ Converted tweets into TF-IDF vectors
- ✅ Trained a **Random Forest Classifier** on 20,000 samples
- ✅ Evaluated accuracy and visualized results
- ✅ Ready to expand into a live web app using Streamlit

---

## 📂 Files in This Repo

| File                      | Description                               |
|--------------------------|-------------------------------------------|
| `sentiment_analysis_app.py` | Main Python script for training & testing |
| `.gitignore`             | Excludes large CSV files from Git         |

---

## 📊 Dataset

- Dataset used: [Sentiment140 Twitter Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- 1.6M tweets labeled as **0 (negative)** or **4 (positive)**

**Note:** The dataset is too large for GitHub, so it’s excluded.  
👉 You can download it from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)  
and rename it as `sentiment_data.csv`.

---

## ⚙️ How to Run This

### 1. Clone the repo
```bash
git clone https://github.com/NaveenKancharla28/sentiment_analysis_app.git
cd sentiment_analysis_app
