import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

st.write("""# Klasifikasi Penguin
         Aplikasi berbasi Web untuk memprediksi jenis penguin **Palmer Penguin**
         """)

img = Image.open('gambar1.png')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

img2 = Image.open('gambar2.png')
img2 = img2.resize((700, 451))
st.image(img2, use_column_width=False)

st.sidebar.header('Parameter Inputan')

# Upload File CSV untuk parameter inputan
upload_file = st.sidebar.file_uploader('Upload file CSV anda', type=['CSV'])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        island = st.sidebar.selectbox(
            'Pulau', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Gender', ('male', 'female'))
        bill_length_mm = st.sidebar.slider(
            'Panjang Paruh (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider(
            'Kedalaman Paruh (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider(
            'Panjang Sirip (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider(
            'Massa Tubuh (g)', 2700.0, 6300.0, 4207.0)
        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex,
        }
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

# Menggabungkan inputan dan dataset penguin
penguin_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguin_raw.drop(columns=['species'])
df = pd.concat([inputan, penguins], axis=0)

# Encode untuk fitur ordinal
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]  # ambil baris pertama (inputan dari user)

# Menampilkan parameter hasil inputan
st.subheader('Parameter Inputan')

if upload_file is not None:
    st.write(df)
else:
    st.write('Menunggu file csv untuk diupload. Saat ini memakai sample inputan')
    st.write(df)


# Load Model NBC
load_model = pickle.load(open('modelNBC_penguin.pkl', 'rb'))

# Menerapkan NBC
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Keterangan Label Kelas')
jenis_penguin = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
jenis = pd.DataFrame({'Nilai': jenis_penguin})
st.write(jenis)

st.subheader('Hasil Prediksi (Klasifikasi Penguin)')
st.write(jenis_penguin[prediksi])

st.subheader('Probabilitas Hasil Prediksi (Klasifikasi Penguin)')
st.write(prediksi_proba)
