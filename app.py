import streamlit as st
import pickle
import string
import nltk
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

# --------------------------
# NLTK Setup
# --------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [ps.stem(i) for i in tokens if i not in stop_words and i not in string.punctuation]
    return " ".join(tokens)

# --------------------------
# Load Model, Vectorizer & Email Data
# --------------------------
@st.cache_resource
def load_model_and_data():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        df_sample = pd.read_csv('balanced.csv', encoding='latin1')  # Ensure it has 'Target' and 'Message'
        df_sample['Message'] = df_sample['Message'].fillna("")
        df_sample['transformed_text'] = df_sample['Message'].apply(transform_text)
        
        # TF-IDF matrix for all emails
        X_text = tfidf.transform(df_sample['transformed_text'])
        return tfidf, model, df_sample, X_text
    except Exception as e:
        st.error(f"❌ Failed to load resources: {e}")
        st.stop()

tfidf, model, df_sample, X_text_all = load_model_and_data()

# --------------------------
# Similarity Score Function
# --------------------------
def similarity_score(email_text):
    vec = tfidf.transform([email_text])
    sims = cosine_similarity(vec, X_text_all)[0]
    # Take the highest similarity across all emails (spam or ham)
    max_idx = np.argmax(sims)
    max_score = sims[max_idx]
    return float(max_score), max_idx

# --------------------------
# Final Weighted Score
# --------------------------
def final_spam_score(email_text):
    vec = tfidf.transform([email_text])
    X_num = np.array([[len(email_text), len(email_text.split())]])
    X_num_sparse = csr_matrix(X_num)
    vec_comb = hstack((vec, X_num_sparse))
    
    ml_prob = model.predict_proba(vec_comb)[0][1]   # ML model probability of spam
    sim_score, sim_idx = similarity_score(email_text)
    
    final_score = 0.6 * ml_prob + 0.4 * sim_score
    return ml_prob, sim_score, final_score, sim_idx

# --------------------------
# Similarity Graph Visualization
# --------------------------
def visualize_similarity(email_text, top_n=30):
    vec = tfidf.transform([email_text])
    sims = cosine_similarity(vec, X_text_all)[0]
    
    # Top N similar emails (spam or ham)
    top_idx = sims.argsort()[-top_n:][::-1]
    
    G = nx.Graph()
    G.add_node("Incoming Email", color='red')
    
    for idx in top_idx:
        sim_value = sims[idx]
        label = f"{'Spam' if df_sample.loc[idx,'Target']==1 else 'Ham'} {idx}"
        G.add_node(label)
        G.add_edge("Incoming Email", label, weight=sim_value)
    
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = [G[u][v]['weight']*5 for u,v in edges]
    
    plt.figure(figsize=(6,4))
    nx.draw(G, pos, with_labels=True, width=weights, node_color='skyblue', font_size=8)
    plt.title("Top Similar Emails (Spam & Ham)")
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# --------------------------
# Streamlit App Layout
# --------------------------
st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Spam Detection System")
st.subheader("Analyze messages for SPAM vs HAM")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📝 Enter Message")
    input_sms = st.text_area("Type here...", height=200)
    analyze_button = st.button('🔍 Analyze Message', use_container_width=True)
    
    if analyze_button:
        if not input_sms.strip():
            st.warning("⚠️ Please enter a message.")
        else:
            transformed_sms = transform_text(input_sms)
            ml_prob, sim_score, final_score, sim_idx = final_spam_score(transformed_sms)
            
            # Determine spam/ham based on weighted score
            result = "SPAM" if final_score >= 0.5 else "HAM"
            
            st.markdown("---")
            if result == "SPAM":
                st.error(f"🚨 SPAM DETECTED — Weighted Score: {final_score:.2f}")
            else:
                st.success(f"✅ NOT SPAM — Weighted Score: {final_score:.2f}")
            
            st.subheader("📊 Scores")
            st.metric("ML Probability", f"{ml_prob*100:.2f}%")
            st.metric("Similarity Score", f"{sim_score*100:.2f}%")
            st.metric("Final Weighted Score", f"{final_score*100:.2f}%")
            
            with st.expander("🔍 View Processed Text"):
                st.code(transformed_sms)
            
            st.subheader("📈 Similarity Graph")
            buf = visualize_similarity(transformed_sms)
            st.image(buf)

with col2:
    st.subheader("ℹ️ How It Works")
    st.info("""
    **Pipeline**
    1. Clean & preprocess text  
    2. Convert to TF-IDF vectors (5000 features)  
    3. Add numerical features (characters + words)  
    4. ExtraTreesClassifier predicts spam/ham  
    5. Compute similarity with ALL emails (spam & ham)  
    6. Weighted final score: 0.6 ML + 0.4 similarity  
    7. Visualize top 5 similar emails (Spam & Ham)
    """)
    st.markdown("---")
    
    st.subheader("📊 Model Details")
    st.write(f"Expected Features: **{model.n_features_in_}**")
    st.write(f"TF-IDF Vocabulary Size: **{len(tfidf.vocabulary_)}**")
    st.write(f"Total Features at Prediction: **{model.n_features_in_ + 2}**")
