import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread

# Função para melhorar a imagem
def melhorar_imagem(img):
    img = img.reshape((1, 256, 256, 3)).astype(float) / 255.
    sub = (modelo.predict(img)).flatten()

    for i in range(len(sub)):
        if sub[i] > 0.5:
            sub[i] = 1
        else:
            sub[i] = 0
    return sub

# Função para aplicar uma máscara na imagem
def aplicar_mascara(img):
    sub = img.reshape((1, 256, 256, 3)).astype(np.float32) / 255.
    mask = np.array(melhorar_imagem(sub).reshape(256, 256), dtype=np.uint8)
    sub2 = img.reshape(256, 256, 3)
    res = cv2.bitwise_and(sub2, sub2, mask=mask)

    return res

# Carregar o modelo
@st.cache
def carregar_modelo():
    return load_model('ResU_net.h5')

modelo = carregar_modelo()

# Barra lateral
with st.sidebar.header('Carregue sua Imagem de Pele'):
    upload_file = st.sidebar.file_uploader('Escolha sua Imagem de Pele', type=['jpg', 'jpeg', 'png'])

# Título da página
st.write('# 🧠Segmentação de Lesões na Pele🧠')
st.write('Este site foi criado por Crinex. O código do site e da segmentação está no Github. Se você deseja usar este código, faça um Fork e use.🤩🤩')
st.write('📕 Github: https://github.com/crinex/Skin-Lesion-Segmentation-Streamlit 📕')

# Tela principal
col1, col2, col3 = st.beta_columns(3)
with col1:
    st.write('### Imagem Original')
    img = imread(upload_file)
    img = resize(img, (256, 256))
    preview_img = resize(img, (256, 256))
    st.image(preview_img)

col2.write('### Botão')
clicked = col2.button('Segmentar!!')
clicked2 = col2.button('Prever Imagem')

if clicked:
    x = img
    x = np.reshape(x, (256, 256, 3))
    col3.write('### Imagem Segmentada')
    mask_img = aplicar_mascara(x)
    col3.image(mask_img)

if clicked2:
    x = img
    x = np.reshape(x, (256, 256, 3))
    enhance_img = melhorar_imagem(x).reshape(256, 256)
    col3.write('### Imagem de Previsão')
    col3.image(enhance_img)
