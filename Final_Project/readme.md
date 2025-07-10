# 🧠 Topic Modeling Web App with KMeans & LDA

A Streamlit-powered web app to perform topic modeling and document clustering on large text corpora using **KMeans** and **Latent Dirichlet Allocation (LDA)**.

---

## 📌 Features

- ✅ Topic modeling using **LDA (Latent Dirichlet Allocation)**
- ✅ Document clustering using **KMeans**
- ✅ Interactive UI built with **Streamlit**
- ✅ Visualizations using **WordClouds**, **PCA plots**, and **top keywords per topic**
- ✅ Dynamic selection of number of clusters/topics
- ✅ Document upload functionality for custom inputs
- ✅ Option to view sample documents by cluster/topic
- ✅ Model summary: vocab size, doc count, silhouette score
- ✅ Fully deployed on **Replit** with public URL access


## ⚙️ How It Works

1. **Preprocessing**
   - Text cleaning: lowercasing, punctuation removal, stopword removal, lemmatization

2. **TF-IDF Vectorization**
   - Converts text into a numerical format for clustering

3. **KMeans**
   - Groups documents into clusters using cosine similarity
   - Visualized using **PCA 2D plots**

4. **LDA (Topic Modeling)**
   - Extracts hidden topics from text
   - Visualized using **WordClouds** and top keywords

---

## 🧪 Tech Stack

- 🐍 Python
- 📊 Scikit-learn
- 🔍 Gensim
- 📄 NLTK
- 🎨 Matplotlib & Seaborn
---

## 🧠 Sample Output

- **Silhouette Score** shown for clustering quality
- **WordClouds** per topic (LDA)
- **Top keywords per cluster** (KMeans)
- **PCA Visualization** of clusters
- **Document previews** inside each cluster/topic

---

## 📁 Folder Structure

```
streamlit-app/
│
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
└── README.md            # You're reading it :)
```

---

## 🚀 How to Run Locally

```bash
# Clone the project
git clone https://github.com/your-username/topic-modeling-app.git
cd topic-modeling-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---


3. Click ✅ **Run**, then **open the generated URL**



## ⭐️ Acknowledgements

- Dataset: [`20 Newsgroups`](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)
- Streamlit documentation
- Gensim & NLTK for LDA modeling
