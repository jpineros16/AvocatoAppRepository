import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import time
import cv2
from tensorflow.keras import preprocessing

fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Clasificador de madurez de aguacate')

st.markdown("Bienvenido. Sube una imagen de un aguacate y te dirá en que estado de maduración se encuentra")


def main():
    file_uploaded = st.file_uploader("Escoge un archivo", type=["png","jpg","jpeg"])
    class_btn = Null
    
    if file_uploaded is not None:    
        imageShow = Image.open(file_uploaded)
        imageBGR = cv2.cvtColor(np.array(imageShow), cv2.COLOR_RGB2BGR)
        imageRescaled = ResizeImage(imageBGR)
        st.image(imageRescaled, caption='Imagen Cargada')#, use_column_width=True)
        class_btn = st.button("Clasificar")
        
    if class_btn:
        if file_uploaded is None:
            st.write("Comando inválido. Por favor sube una imagen")
        else:
            with st.spinner('Modelo trabajando...'):
                predictions = predict(imageBGR)
                st.write("Imagen clasificada con éxito")
                st.success(predictions)                

                
def ResizeImage(imagen):
    imageRGB = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    percentage = 1/4
    width = int(1536*percentage)
    height = int(2048*percentage)
    imageRe = cv2.resize(imageRGB, (width, height))
    return imageRe


def predict(image):
    classifier_model = "simple_NN_model.h5"
    model = load_model(classifier_model, compile=False)
    test_image = cv2.resize(image, (24,32))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/ 255.0

    test_image = test_image.flatten()
    test_image = np.expand_dims(test_image, axis=0)
    
    class_names = [
          'Maduro',
          'Sobremaduro',
          'Verde']
    predictions = model.predict(test_image)
    scores = predictions
   
    result = f"{class_names[np.argmax(scores)]} con { (100 * np.max(scores)).round(2) } % precisión." 
    adc = "Es un aguacate en etapa "
    textoFin = adc + result
    return textoFin



if __name__ == "__main__":
    main()
