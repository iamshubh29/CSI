# 🧠 Topic Modeling and Document Clustering Web App using KMeans & LDA

This is a fully functional Streamlit web app for **unsupervised text clustering and topic modeling**, designed to help analyze and visualize large-scale document corpora using **KMeans** and **Latent Dirichlet Allocation (LDA)**. The project leverages the **20 Newsgroups** dataset to demonstrate both document clustering and latent topic extraction.

---

## 📊 Project Objective

The goal of this project is to:
- Group similar documents into clusters using unsupervised algorithms
- Identify hidden topics using LDA topic modeling
- Visualize and interpret the results interactively
- Allow users to upload custom text files for dynamic analysis

---

## ✅ What We Have Done

### 🔹 1. Dataset Loading
- Loaded the **20 Newsgroups dataset** directly from `scikit-learn.datasets`.
- Limited to the first 2000 documents for performance.

### 🔹 2. Text Preprocessing
- Applied **text cleaning** including:
  - Removal of special characters and punctuation
  - Lowercasing
  - Stopword removal using NLTK
  - Lemmatization using `WordNetLemmatizer`

### 🔹 3. TF-IDF Vectorization
- Converted the cleaned documents into numerical vectors using `TfidfVectorizer`.
- Limited to the top 2000 features for efficiency.

### 🔹 4. Clustering Techniques
Implemented and compared multiple clustering models:
- **KMeans Clustering**
  - Number of clusters adjustable using a slider
  - Top keywords per cluster extracted from centroids
  - Silhouette score computed
- **Agglomerative Hierarchical Clustering** *(optional in notebook)*
- **DBSCAN Clustering**
  - With tuning of `eps` and `min_samples`
  - Auto-detects outliers
- **GMM (Gaussian Mixture Models)** *(explored for soft clustering)*

### 🔹 5. Topic Modeling using LDA
- Built an LDA model using **Gensim** and **Bag of Words** format.
- Extracted top keywords per topic.
- Visualized topics using **WordClouds**.

### 🔹 6. Dimensionality Reduction and Visualization
- Used **PCA** to project TF-IDF vectors into 2D for visualization.
- Colored document points by cluster labels.

### 🔹 7. Model Evaluation
- Computed **Silhouette Scores** for:
  - KMeans
  - DBSCAN (if applicable)
  - Agglomerative Clustering
- Compared scores to determine best-performing clustering technique.

### 🔹 8. Streamlit Web App Development
- Built an interactive UI using **Streamlit** with:
  - Sidebar model selector (`KMeans` or `LDA`)
  - Cluster/topic number sliders
  - Visuals: PCA plots, WordClouds, top terms
  - Sample documents per topic shown
  - Upload feature for user files (CSV/TXT)

---

## 🖼 Screenshots & Visualizations

- ✅ PCA Scatter Plot for KMeans Clusters
- ✅ WordClouds for LDA Topics
- ✅ Top Keywords per Cluster/Topic
- ✅ Silhouette Scores Displayed in Real-Time
- ✅ Sample Documents Preview per Group


## 📂 Project Structure

```
streamlit-topic-modeling/
│
├── app.py               # Streamlit app source code
├── requirements.txt     # All Python dependencies
├── runtime.txt          # Python version pinning for compatibility
└── README.md            # Project documentation (this file)
```

---

## 🧠 Technologies Used

| Component         | Library Used             |
|------------------|--------------------------|
| Web Framework     | Streamlit                |
| NLP Preprocessing | NLTK                     |
| Topic Modeling    | Gensim (LDA)             |
| Clustering        | scikit-learn (KMeans, DBSCAN, Agglomerative) |
| Vectorization     | TfidfVectorizer          |
| Visualization     | Matplotlib, Seaborn, WordCloud, PCA |
| Dataset           | 20 Newsgroups (Sklearn)  |

---

## ⚙️ Local Setup

### 📦 1. Clone the Repository

```bash
git clone https://github.com/your-username/topic-modeling-streamlit.git
cd topic-modeling-streamlit
```

### 🧪 2. Create Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### ▶️ 3. Run Streamlit App

```bash
streamlit run app.py
```

---

## 🌟 Acknowledgements

- [scikit-learn 20 Newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)
- [Streamlit](https://streamlit.io)
- [Gensim](https://radimrehurek.com/gensim/)
- [NLTK](https://www.nltk.org/)
