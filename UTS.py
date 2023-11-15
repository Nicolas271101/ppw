import streamlit as st
import requests
from bs4 import BeautifulSoup
import csv
import base64
import pandas as pd
import numpy as np
import re, string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from itertools import chain
import gensim
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download('corpus')

tab1, tab2, tab3, tab4, tab5, tab6= st.tabs(["DataSet", "Cleaning", 'Tokenize', 'Stopword', 'Topik dalam dokumen', 'Kata dalam Topik'])


with tab1 :
    csv_path = 'https://raw.githubusercontent.com/Nicolas271101/ppw/main/dataptapgsd.csv'
    df = pd.read_csv(csv_path)
    df

with tab2 : 
    # Text Cleaning
    def cleaning(text):
    # HTML Tag Removal
        text = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(text))

    # Case folding(mengecilkan )
        text = text.lower()

    # Trim text (menghilangkan spasi depan dan belakang)
        text = text.strip()

    # Remove punctuations, karakter spesial, and spasi ganda
        text = re.compile('<.*?>').sub('', text)
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
        text = re.sub('\s+', ' ', text)

        # Number removal
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Mengubah text 'nan' dengan whitespace agar nantinya dapat dihapus
        text = re.sub('nan', '', text)

        return text
    # Ubah empty string menjadi NaN value
    df = df.replace('', np.nan)
    # Cek missing values
    df.isnull().sum()
    # Remove missing values
    df.dropna(inplace=True)

    df=df.astype(str)
    df["abstrak"] = df["abstrak"].apply(lambda x: cleaning(x))

    abstrak_column = df["abstrak"]

    df["abstrak"]

with tab3 :
    # Tokenizing Abstrak
    df['processed_abstrak'] = df['abstrak'].apply(lambda x: word_tokenize(x))
    df[["processed_abstrak", "abstrak"]]


with tab4 :
    stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))

    df['processed_abstrak'] = df['processed_abstrak'].apply(lambda x: [w for w in x if not w in stop_words])

    df[['processed_abstrak', 'abstrak']]


with tab5 :

    # Ubah teks ke dalam format yang cocok untuk Gensim
    documents =df['processed_abstrak']

    # Membuat kamus (dictionary) dari kata-kata unik dalam dokumen
    dictionary = corpora.Dictionary(documents)

    # Membuat korpus (bag-of-words) dari dokumen
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Melatih model LDA
    lda_model = LdaModel(corpus, num_topics=6, id2word=dictionary, passes=30)

    # Membuat DataFrame untuk menampilkan proporsi topik dalam dokumen
    document_topic_df = pd.DataFrame()
    judul = df['Judul']

    for doc in corpus:
        topic_distribution = lda_model.get_document_topics(doc, minimum_probability=0)
        # doc_topic_props = {f"Topik {topic_id + 1}": prop for topic_id, prop in topic_distribution}
        doc_topic_props = {} #mengubah tampilan agar topik di probalility hilang dan ada pada tabel diatasnya
        for topic_id, prob in topic_distribution:
            key = f"Topik {topic_id + 1}"
            doc_topic_props[key] = prob
        document_topic_df = pd.concat([document_topic_df, pd.Series(doc_topic_props)], ignore_index= True, axis=1)

    # document_topic_df ['Judul'] = judul
    document_topic_df = document_topic_df.transpose()  # Transpose agar topik menjadi kolom

    # document_topic_df.insert(document_topic_df.columns.get_loc('Topik 1') , 'Judul', judul)
    document_topic_df ['Judul'] = judul
    # Menampilkan tabel proporsi topik dalam dokumen
    document_topic_df
    def simpanlda(gabung):
        csv = gabung.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="Hasil_LDA.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    if st.button("Download LDA"):
        simpanlda(document_topic_df)
    
    def simpanclus(gabung):
        csv = gabung.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="Hasil_Clustering.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    st.header("Bobot Topik pada Dokumen untuk Clustering")
    topics = pd.DataFrame(document_topic_df, columns=['Topik 1','Topik 2','Topik 3', 'Topik 4', 'Topik 5', 'Topik 6'])
    gabung = pd.concat([judul, topics], axis=1)
    banyak = st.number_input('Masukkan angka', 1, 10)
    if st.button('Lakukan Klastering'):
        num_clusters = banyak
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
        clusters = kmeans.fit_predict(topics)
        topics['Cluster'] = clusters
        mebel = pd.concat([gabung, topics['Cluster']], axis=1)
        st.dataframe(mebel)
    if st.button('Download Clustering'):
        simpanclus(mebel)
    

with tab6 :
    # Membuat DataFrame untuk menampilkan proporsi kata dalam topik
    topic_word_df = pd.DataFrame()

    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic (topic_id , topn=10)  # Ambil 10 kata kunci teratas
        # # words_list = [word for word, _ in topic_words]
        words_list = []
        for word, bbt in topic_words:
            words_list.append(word)
        topic_word_df[f"Topik {topic_id + 1}"] = words_list

    # Menampilkan tabel proporsi kata dalam topik
    topic_word_df
