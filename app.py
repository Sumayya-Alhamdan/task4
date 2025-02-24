import streamlit as st
import nltk
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Set NLTK data directory to avoid missing resources
os.environ['NLTK_DATA'] = './nltk_data'

# Ensure necessary NLTK resources are downloaded
nltk.download('reuters', download_dir='./nltk_data')
nltk.download('punkt', download_dir='./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')

# Load and preprocess Reuters corpus
@st.cache_data
def load_corpus():
    corpus_sentences = []
    for fileid in reuters.fileids():
        raw_text = reuters.raw(fileid)
        tokenized_sentence = [word for word in word_tokenize(raw_text) if word.isalnum()]
        corpus_sentences.append(tokenized_sentence)
    return corpus_sentences

st.title("Word Embedding Explorer")

# Load corpus and train Word2Vec model
corpus_sentences = load_corpus()
model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)

# User input for word similarity
word = st.text_input("Enter a word to find similar words:")
if st.button("Find Similar Words"):
    if word in model.wv:
        similar_words = model.wv.most_similar(word, topn=10)
        st.write("### Most Similar Words:")
        for w, score in similar_words:
            st.write(f"- **{w}** (Score: {score:.4f})")
    else:
        st.write("Word not in vocabulary.")

# Visualization using t-SNE
if st.button("Visualize Embeddings"):
    words = list(model.wv.index_to_key)[:200]
    word_vectors = np.array([model.wv[word] for word in words])
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(word_vectors)
    
    fig, ax = plt.subplots()
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
    for i, word in enumerate(words):
        ax.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=8)
    
    st.pyplot(fig)
