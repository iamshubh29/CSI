import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import gensim
from gensim import corpora
from wordcloud import WordCloud
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(layout="wide")
st.title("🧠 Topic Modeling with KMeans and LDA")
st.markdown("Upload your own text data or use the built-in 20 Newsgroups dataset.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload a `.csv`, `.txt`, or `.xlsx` file", type=["csv", "txt", "xlsx"])

# Load default sample
@st.cache_data
def load_default_sample():
    data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'), subset='all')
    return data.data[:2000]

# Load user or default documents
if uploaded_file:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        documents = df.iloc[:, 0].dropna().astype(str).tolist()
    elif file_name.endswith(".txt"):
        documents = uploaded_file.read().decode("utf-8").splitlines()
        documents = [doc.strip() for doc in documents if doc.strip()]
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        documents = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        st.error("Unsupported file format.")
        documents = []
else:
    documents = load_default_sample()

# Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 3]
    return " ".join(tokens)

start_time = time.time()
with st.spinner("Preprocessing documents..."):
    cleaned_docs = [preprocess_text(doc) for doc in documents]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(cleaned_docs)

# Dataset summary
st.subheader("📊 Dataset Summary")
st.metric("Document Count", len(documents))
st.metric("Vocabulary Size", len(vectorizer.get_feature_names_out()))
st.metric("Preprocessing Time (s)", f"{time.time() - start_time:.2f}")

# Tabs for model results
tab1, tab2 = st.tabs(["KMeans Clustering", "LDA Topic Modeling"])

# ----------------------------- KMEANS -----------------------------
with tab1:
    st.header("📌 KMeans Clustering")
    k = st.slider("Select number of clusters (K)", 2, 20, 5)

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_tfidf)
    silhouette = silhouette_score(X_tfidf, kmeans_labels)
    st.markdown(f"**Silhouette Score:** `{silhouette:.4f}`")

    # PCA visualization
    st.subheader("📉 PCA Cluster Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tfidf.toarray())
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="tab10", s=10)
    ax.set_title("PCA Plot of KMeans Clusters")
    st.pyplot(fig)

    # Top terms + samples per cluster
    st.subheader("🧠 Cluster Details")
    terms = vectorizer.get_feature_names_out()
    for i in range(k):
        with st.expander(f"🔹 Cluster {i}"):
            top_terms = [terms[ind] for ind in kmeans.cluster_centers_[i].argsort()[-10:][::-1]]
            st.markdown(f"**Top Terms:** {', '.join(top_terms)}")

            sample_docs = [doc for j, doc in enumerate(cleaned_docs) if kmeans_labels[j] == i][:3]
            for idx, doc in enumerate(sample_docs):
                st.markdown(f"**Sample {idx+1}:** {doc[:300]}...")

# ----------------------------- LDA -----------------------------
with tab2:
    st.header("📌 Latent Dirichlet Allocation (LDA)")
    n_topics = st.slider("Select number of topics", 2, 20, 5)

    tokenized_docs = [doc.split() for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        passes=10
    )

    # Topic list
    st.subheader("🧾 Topics")
    for idx, topic in lda_model.print_topics(num_words=10):
        with st.expander(f"🟠 Topic {idx}"):
            st.markdown(topic)

    # WordClouds
    st.subheader("🌥️ WordClouds for Topics")
    for t in range(n_topics):
        fig, ax = plt.subplots()
        wordcloud = WordCloud(background_color="white").fit_words(dict(lda_model.show_topic(t, 30)))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
