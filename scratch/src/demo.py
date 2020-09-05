import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.backend import get_session

def main():
    st.write('# Realtor.com Image Synthesis Demo')
    st.write('')
    st.write('')
    st.write('')

    gans1, gans2, session = load_models()
    set_session(session)
    cities = ['Boston', 'Charleston', 'Chicago', 'Denver', 'Houston', 'LosAngeles', 'Miami', 'Morgantown', 'Portland', 'Seattle']
    city = st.sidebar.selectbox('Select city', cities)
    img_types = ['exterior', 'landscape']
    img_type = st.sidebar.selectbox('Select type', ['exterior', 'landscape'])

    for i in range(len(cities)):
        if cities[i] == city:
            label = i

    if img_type == 'exterior':
        generator = gans1
    if img_type == 'landscape':
        generator = gans2

    noise = tf.random.normal([1, 100])
    if st.sidebar.checkbox('Set seed'):
        seed = st.sidebar.number_input('', 1)
        tf.random.set_seed(seed)
        noise = tf.random.normal([1, 100])
    generated_image = generator([noise,np.array([label])], training=False)
    img_array = generated_image[0, :, :].numpy()
    img_array = (img_array - np.min(img_array))/np.ptp(img_array)

    if st.sidebar.button('Generate image'):
        st.image(img_array, width = 500)
        st.write('Right-click on the image to save')

@st.cache(allow_output_mutation=True)
def load_models():
    gans1 = load_model('generator_large_ext.h5', compile = False)
    gans2 = load_model('generator_large_land.h5', compile = False)
    session = get_session()
    return gans1, gans2, session

if __name__ == "__main__":
    main()