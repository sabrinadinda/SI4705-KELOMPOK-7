
import streamlit as st
import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.models import CoherenceModel
import plotly.express as px
import joblib

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Dashboard LDA Topic Modeling",
    page_icon="📊",
)

# Judul aplikasi
st.title("LDA Topic Modeling Dashboard")
st.write("Analisis topik dari dataset BBC menggunakan model LDA.")

st.markdown("---")
st.subheader("Hasil Analisis Topik")

# ====================== 1. Load dan Persiapkan Data ======================
@st.cache_data
def load_data():
    data = pd.read_csv("bbc_cleaned.csv")
    data = data.dropna(subset=['description'])
    return data

data = load_data()

# ====================== 2. Preprocessing dan Model LDA ======================
@st.cache_resource
def train_lda_model():
    # Preprocessing
    def preprocess(text):
        return [word for word in simple_preprocess(text, deacc=True) if word not in STOPWORDS]

    data['tokens'] = data['description'].apply(preprocess)
    
    # Dictionary dan Corpus
    dictionary = corpora.Dictionary(data['tokens'])
    corpus = [dictionary.doc2bow(text) for text in data['tokens']]
    
    # Latih model LDA
    lda_model = models.LdaModel(corpus=corpus,
                                id2word=dictionary,
                                num_topics=5,
                                random_state=42,
                                passes=15,
                                alpha='auto',
                                per_word_topics=True)
    
    # Simpan model (opsional)
    joblib.dump(lda_model, 'lda_model.pkl')
    joblib.dump(dictionary, 'dictionary.pkl')
    return lda_model, dictionary, corpus, data

lda_model, dictionary, corpus, data = train_lda_model()

# ====================== 3. Fungsi untuk Persentase Term ======================
def get_term_percentage(lda_model, num_topics, num_words=5):
    term_weights = {i: {} for i in range(num_topics)}
    for topic_id in range(num_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=num_words)
        for term, weight in topic_terms:
            term_weights[topic_id][term] = weight
    total_weights = {i: sum(term_weights[i].values()) for i in range(num_topics)}
    term_percentage = {i: {} for i in range(num_topics)}
    for topic_id in range(num_topics):
        for term, weight in term_weights[topic_id].items():
            term_percentage[topic_id][term] = (weight / total_weights[topic_id]) * 100
    return term_percentage

term_percentage = get_term_percentage(lda_model, num_topics=5, num_words=5)

# ====================== 4. Komponen Dashboard ======================
# Dropdown untuk memilih topik
st.subheader("Pilih Topik")
topic_id = st.selectbox("Pilih Topik untuk Ditampilkan:", [f"Topik {i+1}" for i in range(5)], index=0)
selected_topic = int(topic_id.split()[1]) - 1

# Tampilkan kata kunci topik
st.subheader("Kata Kunci Topik")
topic_terms = lda_model.show_topic(selected_topic, topn=5)
keywords = ", ".join([f"{term} ({weight:.3f})" for term, weight in topic_terms])
st.write(f"Kata kunci untuk {topic_id}: {keywords}")

# Tampilkan persentase term
st.subheader("Persentase Term per Topik")
terms = list(term_percentage[selected_topic].keys())
percentages = list(term_percentage[selected_topic].values())
for term, percentage in zip(terms, percentages):
    st.write(f"{term}: {percentage:.2f}%")

# Visualisasi persentase term
st.subheader("Visualisasi Persentase Term")
fig = px.bar(
    x=terms,
    y=percentages,
    labels={'x': 'Term', 'y': 'Persentase (%)'},
    title=f'Persentase Term pada {topic_id}',
    color=terms,
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ====================== 5. Informasi Tambahan ======================
st.markdown("---")
st.subheader("Informasi Dataset")
st.write(f"Jumlah dokumen: {len(data)}")
st.write("Dataset: bbc_cleaned.csv")
st.write("Jumlah topik: 5")
st.write("Jumlah kata kunci per topik: 5")

# ====================== 6. Evaluasi Model LDA ======================
st.markdown("---")
st.subheader("📈 Evaluasi Model LDA")

# Opsi evaluasi
with st.expander("⚙️ Opsi Evaluasi"):
    coherence_type = st.selectbox("Pilih Jenis Coherence", ['c_v', 'u_mass'], index=0)
    max_docs = st.slider("Jumlah dokumen untuk evaluasi (coherence)", min_value=100, max_value=len(corpus), value=min(1000, len(corpus)), step=100)

# Fungsi menghitung topic diversity
def topic_diversity(model, topn=10):
    topics = model.show_topics(num_topics=model.num_topics, num_words=topn, formatted=False)
    all_words = [word for topic in topics for word, _ in topic[1]]
    unique_words = set(all_words)
    return len(unique_words) / (model.num_topics * topn)

# Evaluasi dengan spinner
with st.spinner("Sedang menghitung metrik evaluasi..."):
    # 1. Perplexity
    perplexity = np.exp2(-lda_model.log_perplexity(corpus))

    # 2. Coherence Score
    sample_texts = data['tokens'][:max_docs]
    sample_corpus = corpus[:max_docs]
    coherence_model = CoherenceModel(model=lda_model, texts=sample_texts, dictionary=dictionary, coherence=coherence_type)
    coherence_score = coherence_model.get_coherence()

    # 3. Log Likelihood
    log_likelihood = lda_model.log_perplexity(corpus) * sum(len(doc) for doc in corpus)

    # 4. Topic Diversity
    diversity = topic_diversity(lda_model)

# Tampilkan hasil evaluasi
col1, col2 = st.columns(2)
with col1:
    st.metric("Perplexity", f"{perplexity:.2f}")
    st.metric("Coherence Score", f"{coherence_score:.2f} ({coherence_type})")
with col2:
    st.metric("Topic Diversity", f"{diversity:.2f}")
    st.metric("Log Likelihood", f"{log_likelihood:.2f}")

# ====================== 7. Input Dataset Baru & Validasi ======================
st.markdown("---")
st.subheader("📤 Validasi Dataset Baru")

uploaded_file = st.file_uploader("Unggah file CSV untuk validasi", type=["csv"])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    if 'description' not in new_data.columns:
        st.error("File harus memiliki kolom 'description'.")
    else:
        # Preprocessing
        def preprocess(text):
            return [word for word in simple_preprocess(text, deacc=True) if word not in STOPWORDS]

        new_data['tokens'] = new_data['description'].apply(preprocess)
        new_corpus = [dictionary.doc2bow(text) for text in new_data['tokens']]
        
        if st.button("Validasi Dataset"):
            topic_results = []
            for bow in new_corpus:
                topic_dist = lda_model.get_document_topics(bow)
                topic_results.append(sorted(topic_dist, key=lambda x: -x[1])[0][0])  # top topic
            
            # ✅ Tambahkan +1 agar topik dari 1–5
            new_data['Topik'] = [topic + 1 for topic in topic_results]

            st.success("Validasi selesai. Berikut hasil topiknya:")
            st.dataframe(new_data[['description', 'Topik']])
