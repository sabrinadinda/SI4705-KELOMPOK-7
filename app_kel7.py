# ====================== 1. Import Library ======================
import pandas as pd
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from gensim import corpora, models

# ====================== 2. Load Dataset ======================
df = pd.read_csv("bbc_cleaned.csv")

# Pastikan kolom 'description' tidak kosong
df = df.dropna(subset=['description'])

# ====================== 3. Preprocessing ======================
def preprocess(text):
    return [word for word in simple_preprocess(text, deacc=True) if word not in STOPWORDS]

df['tokens'] = df['description'].apply(preprocess)

# ====================== 4. Dictionary dan Corpus ======================
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]

# ====================== 5. LDA Topic Modeling ======================
lda_model = models.LdaModel(corpus=corpus,
                            id2word=dictionary,
                            num_topics=5,         # Ubah jumlah topik jika diperlukan
                            random_state=42,
                            passes=15,
                            alpha='auto',
                            per_word_topics=True)

# ====================== 6. Tampilkan Topik ======================
topics = lda_model.print_topics(num_words=5)
for idx, topic in topics:
    print(f"Topik {idx + 1}:\n{topic}\n")


# ====================== 7. Hitung Persentase Term per Topik ======================
def get_term_percentage(lda_model, corpus, dictionary, num_topics, num_words=5):
    # Inisialisasi dictionary untuk menyimpan bobot term per topik
    term_weights = {i: {} for i in range(num_topics)}
    
    # Ambil bobot term dari model LDA
    for topic_id in range(num_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=num_words)
        for term, weight in topic_terms:
            term_weights[topic_id][term] = weight
    
    # Hitung total bobot per topik
    total_weights = {i: sum(term_weights[i].values()) for i in range(num_topics)}
    
    # Hitung persentase kontribusi term per topik
    term_percentage = {i: {} for i in range(num_topics)}
    for topic_id in range(num_topics):
        for term, weight in term_weights[topic_id].items():
            term_percentage[topic_id][term] = (weight / total_weights[topic_id]) * 100
    
    return term_percentage

# ====================== 8. Tampilkan Persentase Term ======================
term_percentage = get_term_percentage(lda_model, corpus, dictionary, num_topics=5, num_words=5)

for topic_id in range(5):
    print(f"\nPersentase Term untuk Topik {topic_id + 1}:")
    for term, percentage in term_percentage[topic_id].items():
        print(f"{term}: {percentage:.2f}%")

# ====================== 9. Evaluasi Model LDA ======================
from gensim.models import CoherenceModel
import numpy as np

# 1. Perplexity
perplexity = lda_model.log_perplexity(corpus)
print(f"Perplexity: {np.exp2(-perplexity):.2f}")

# 2. Coherence Score
coherence_model = CoherenceModel(model=lda_model, texts=df['tokens'], dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model.get_coherence()
print(f"Coherence Score: {coherence_score:.2f}")

# 3. Log Likelihood
log_likelihood = lda_model.log_perplexity(corpus) * sum(len(doc) for doc in corpus)
print(f"Log Likelihood: {log_likelihood:.2f}")

# 4. Topic Diversity
def topic_diversity(model, topn=10):
    topics = model.show_topics(num_topics=model.num_topics, num_words=topn, formatted=False)
    all_words = [word for topic in topics for word, _ in topic[1]]
    unique_words = set(all_words)
    return len(unique_words) / (model.num_topics * topn)

diversity = topic_diversity(lda_model)
print(f"Topic Diversity: {diversity:.2f}")


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

# ====================== CLUSTERING ======================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Kolom numerik untuk clustering
features = ['title_length', 'description_length', 'category_length', 'published_length']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# Dictionary konfigurasi cluster
cluster_configs = {
    3: {
        'model': KMeans(n_clusters=3, random_state=42),
        'label_map': {
            0: 'Berita Pendek',
            1: 'Berita Sedang',
            2: 'Berita Panjang',
        },
    },
    4: {
        'model': KMeans(n_clusters=4, random_state=42),
        'label_map': {
            0: 'Sangat Pendek',
            1: 'Pendek',
            2: 'Sedang',
            3: 'Panjang',
        },
    },
    5: {
        'model': KMeans(n_clusters=5, random_state=42),
        'label_map': {
            0: 'Sangat Pendek',
            1: 'Pendek',
            2: 'Sedang',
            3: 'Panjang',
            4: 'Sangat Panjang',
        },
    },
    6: {
        'model': KMeans(n_clusters=6, random_state=42),
        'label_map': {
            0: 'Sangat Pendek',
            1: 'Pendek',
            2: 'Agak Pendek',
            3: 'Sedang',
            4: 'Panjang',
            5: 'Sangat Panjang',
        },
    }
}

# Proses semua cluster
for n_cluster, config in cluster_configs.items():
    model = config['model']
    label_map = config['label_map']
    
    cluster_col = f'Cluster_{n_cluster}'
    label_col = f'Kategori_Berita_{n_cluster}'
    
    data[cluster_col] = model.fit_predict(scaled_features)
    
    # Remap cluster ke urutan berdasarkan title_length
    avg_title_length = data.groupby(cluster_col)['title_length'].mean().sort_values()
    remap = {old: new for new, old in enumerate(avg_title_length.index)}
    data[cluster_col] = data[cluster_col].map(remap)
    
    # Apply label kategori
    data[label_col] = data[cluster_col].map(label_map)

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

# ====================== Dashboard Clustering ======================
import streamlit as st
import matplotlib.pyplot as plt

st.markdown("---")
st.subheader("🧠 Clustering Panjang Berita")

# Dropdown pilihan cluster
selected_cluster = st.selectbox("Pilih jumlah cluster", [3, 4, 5, 6], index=0)

# Tentukan nama kolom berdasarkan pilihan
cluster_col = f'Cluster_{selected_cluster}'
label_col = f'Kategori_Berita_{selected_cluster}'

st.write(f"**Tampilan hasil clustering untuk {selected_cluster} kategori:**")
st.dataframe(data[['title', 'title_length', cluster_col, label_col]].head(10))

# Visualisasi distribusi
st.write("Distribusi jumlah berita per kategori:")
category_counts = data[label_col].value_counts().sort_index()
fig, ax = plt.subplots()
category_counts.plot(kind='bar', ax=ax, color='lightgreen')
ax.set_title(f"Distribusi Kategori Berita (Cluster {selected_cluster})")
ax.set_xlabel("Kategori")
ax.set_ylabel("Jumlah Berita")
st.pyplot(fig)

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

from pyngrok import ngrok
import threading
import time
import os

def run_streamlit():
    os.system("streamlit run app.py --server.port 8501")

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

time.sleep(5)

public_url = ngrok.connect(addr=8501)
print(f"Streamlit app is live at: {public_url}")