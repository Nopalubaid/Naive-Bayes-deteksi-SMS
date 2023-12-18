import pickle
import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import pytesseract
from PIL import Image
# from googletrans import Translator  # Import Translator from googletrans library
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import altair as alt

# Fungsi untuk menampilkan halaman 2
def show_other_page():
    
    # Load dataset
    st.title('Preprosesing')
    # Splash Screen
    with st.spinner("Sedang memuat..."):
        time.sleep(2)
    st.write('dataset yang digunakan')
    data = pd.read_csv("D:\kuliah nopal\semester 3\Data Mining\dataset crawling\deteksi spam\dataset_sms.csv")
    st.dataframe(data)

    # Load nltk stopwords for Indonesian
    stopwords_ind = stopwords.words('indonesian')

    # Add more stopwords
    more_stopwords = ['tsel', 'gb', 'rb', 'btw']
    stopwords_ind += more_stopwords

    # Initialize Sastrawi stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Fungsi case folding
    def casefolding(text):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[-+]?[0-9]+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.strip()
        return text

    # Fungsi normalisasi kata
    def text_normalize(text, key_norm):
        text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0]
                         if (key_norm['singkat'] == word).any()
                         else word for word in text.split()
                         ])

        text = str.lower(text)
        return text

    # Fungsi stopwords removal
    def remove_stop_word(text):
        clean_words = [word for word in word_tokenize(text) if word.lower() not in stopwords_ind]
        return " ".join(clean_words)

    # Fungsi stemming
    def stemming(text):
        return stemmer.stem(text)

    # Fungsi TF-IDF vectorization
    def tfidf_vectorization(corpus):
        vec_TF_IDF = TfidfVectorizer(ngram_range=(1, 1))
        vec_TF_IDF.fit(corpus)

        # Save the TF-IDF model and vocabulary
        pickle.dump(vec_TF_IDF, open("tfidf_model.pkl", "wb"))
        pickle.dump(vec_TF_IDF.vocabulary_, open("feature_tf-idf.pkl", "wb"))

        return vec_TF_IDF.transform(corpus)

    # Load key_norm dataset
    key_norm_path = 'D:\kuliah nopal\semester 3\Data Mining\dataset crawling\deteksi spam\key_norm.csv'
    key_norm = pd.read_csv(key_norm_path)

    # Case folding and word normalization comparison
    st.subheader("Preprocessing Comparison:")

    # Select a sample index
    sample_index = st.slider("Select data index", 0, len(data) - 1, 111)

    # Get raw, case-folded, word-normalized, stopwords-removed, and stemmed data
    raw_data = data['teks'].iloc[sample_index]
    case_folding = casefolding(raw_data)
    word_normal = text_normalize(case_folding, key_norm)
    stopwords_removal = remove_stop_word(word_normal)
    text_stemming = stemming(stopwords_removal)

    # Display raw, case-folded, word-normalized, stopwords-removed, and stemmed data
    st.write('Raw Data\t :', raw_data)
    st.write('Case Folding\t :', case_folding)
    st.write('Word Normalize\t :', word_normal)
    st.write('Stopwords Removal\t :', stopwords_removal)
    st.write('Stemming\t :', text_stemming)

    # Hasil preprosesing
    st.subheader('dataset hasil preprosesing')
    hasil = pd.read_csv("D:\kuliah nopal\semester 3\Data Mining\dataset crawling\deteksi spam\clean_data.csv")
    st.dataframe(hasil)

    # TF-IDF vectorization
    st.subheader("TF-IDF Vectorization:")
    # st.write("Columns in 'hasil' DataFrame:")
    # st.write(hasil.columns)

    x = hasil['clean_teks']
    y = hasil['label']

    # Display feature (x) and target (y)
    st.write("Memisah kolom feature dan target")
    st.write("Feature (x):")
    st.write(x)

    st.write("Target (y):")
    st.write(y)

    x = x.fillna('')  # Replace NaN with an empty string
    # Unigram
    vec_TF_IDF = TfidfVectorizer(ngram_range=(1, 1))
    x_tf_idf = vec_TF_IDF.fit_transform(x)

    # Display TF-IDF results
    st.write("TF-IDF Vectorized Feature (x_tf_idf):")
    st.write(x_tf_idf)

    # Train-test split for Naive Bayes classification
    # x_train, x_test, y_train, y_test = train_test_split(x_tf_idf, y, test_size=0.2, random_state=0)

    # # Naive Bayes model
    # model = MultinomialNB()
    # model.fit(x_train, y_train)

    # # Predictions and evaluation
    # st.subheader("Naive Bayes Classification Results:")
    # predicted = model.predict(x_test)

    # # Confusion matrix and classification report
    # CM = confusion_matrix(y_test, predicted)
    # st.write("Confusion Matrix:")
    # st.write(CM)

    # st.write("Classification Report:")
    # st.write(classification_report(y_test, predicted))

################################################################################################################################################
################################################################################################################################################

# Fungsi untuk menampilkan halaman 1
def show_home():
    # LOAD SAVE MODEL
    model_fraud = pickle.load(open('D:\kuliah nopal\semester 3\Data Mining\dataset crawling\deteksi spam\model_fraud.sav', 'rb'))

    # Load TF-IDF vectorizer with vocabulary
    loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("D:\kuliah nopal\semester 3\Data Mining\dataset crawling\deteksi spam\qnew_selected_feature_tf-idf.sav", "rb"))))

    # Set page title and description
    st.title('Prediksi Pesan Penipuan')
    st.write('Aplikasi ini digunakan untuk mendeteksi apakah Pesan adalah penipuan atau tidak.')

    # Splash Screen
    with st.spinner("Sedang memuat..."):
        time.sleep(2)

    # Create a sidebar for additional options or information
    # st.sidebar.header('Tentang Aplikasi')
    # st.sidebar.write('Aplikasi ini menggunakan model Text Mining untuk mendeteksi penipuan dalam Pesan.')
    # st.sidebar.write('By Muhammad Naufal Ubaidillah - A11.2022.14408')
    # navigation_options = ["Beranda", "proses"]
    # page_selection = st.sidebar.selectbox("Pilih Halaman", navigation_options)
    # # Menampilkan halaman berdasarkan pilihan navigasi
    # if page_selection == "Beranda":
    #     show_home()
    # elif page_selection == "proses":
    #     show_other_page()

    # Create a text input field for user input
    clean_text = st.text_area('Masukkan Teks Pesan', '')

    # Create a file uploader for images
    # uploaded_image = st.file_uploader("Upload Gambar Pesan", type=["jpg", "png", "jpeg"])

    # Create buttons for prediction and translation
    if st.button('Deteksi Penipuan'):
        if clean_text:
            loaded_vec.fit([clean_text])
            predict_fraud = model_fraud.predict(loaded_vec.transform([clean_text]))

            if predict_fraud == 0:
                fraud_detection = 'Pesan Normal'
                predict_label = 0
            elif predict_fraud == 1:
                fraud_detection = 'Pesan Penipuan'
                predict_label = 1
            else:
                fraud_detection = 'Pesan Promo'
                predict_label = 2

            st.subheader('Hasil Deteksi Teks:')
            st.write(fraud_detection)

            # Calculate and display classification report
            # st.subheader('Metrik Klasifikasi:')
            # report = classification_report([predict_label], predict_fraud, labels=[0, 1, 2])
            # st.text(report)
            
            # Calculate and display accuracy
            # accuracy = accuracy_score([predict_label], predict_fraud)
            # st.subheader('Akurasi:')
            # st.write(accuracy)

            # Calculate and display classification report
            st.subheader('Metrik Klasifikasi:')
            report_dict = classification_report([predict_label], predict_fraud, labels=[0, 1, 2], output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            # Melt the DataFrame for better visualization
            melted_df = pd.melt(report_df.reset_index(), id_vars=['index'], var_name='metric', value_name='score')

            # Display the classification report as a bar chart
            chart = alt.Chart(melted_df).mark_bar().encode(
                x='index:N',
                y='score:Q',
                color='metric:N',
                column='metric:N'
            ).configure_mark(color='blue')

            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning('Masukkan teks Pesan terlebih dahulu.')

# if st.button('Terjemahkan Gambar'):
#     if uploaded_image:
#         # Use Tesseract to extract text from the uploaded image
#         image = Image.open(uploaded_image)
#         text_from_image = pytesseract.image_to_string(image)

#         st.subheader('Hasil Deteksi Teks dari Gambar Pesan:')
#         st.write(text_from_image)

#         # Translate the text to English
#         translator = Translator()
#         translated_text = translator.translate(text_from_image, src='auto', dest='en')

#         #st.subheader('Hasil Terjemahan:')
#         #st.write(translated_text.text)  # Display the translated text

#         # Deteksi penipuan pada teks yang diterjemahkan
#         loaded_vec.fit([translated_text.text])
#         predict_fraud = model_fraud.predict(loaded_vec.transform([translated_text.text]))[0]

#         if predict_fraud == 0:
#             fraud_detection = 'Pesan Normal'
#         elif predict_fraud == 1:
#             fraud_detection = 'Pesan Penipuan'
#         else:
#             fraud_detection = 'Pesan Promo'

       

#         # Menerjemahkan teks ke bahasa Indonesia jika asal bahasa Inggris
#         if translated_text.src == 'en':
#             translator_id = Translator()
#             translated_to_indonesian = translator_id.translate(translated_text.text, src='en', dest='id')

#             st.subheader('Terjemahan ke Bahasa Indonesia:')
#             st.write(translated_to_indonesian.text)

#         # Menerjemahkan teks ke bahasa Inggris jika asal bahasa Indonesia
#         elif translated_text.src == 'id':
#             translator_en = Translator()
#             translated_to_english = translator_en.translate(translated_text.text, src='id', dest='en')

#             st.subheader('Terjemahan ke Bahasa Inggris:')
#             st.write(translated_to_english.text)

#     st.subheader('Hasil Deteksi Penipuan:')
#     st.write(fraud_detection)

# Pemanggilan fungsi untuk menampilkan halaman
navigation_options = ["Beranda", "proses"]
page_selection = st.sidebar.selectbox("Pilih Halaman", navigation_options)

# Menampilkan halaman berdasarkan pilihan navigasi
if page_selection == "Beranda":
    show_home()
elif page_selection == "proses":
    show_other_page()

st.sidebar.header('Tentang Aplikasi')
st.sidebar.write('Aplikasi ini menggunakan model Text Mining untuk mendeteksi penipuan dalam Pesan.')
st.sidebar.write('By Muhammad Naufal Ubaidillah - A11.2022.14408')
