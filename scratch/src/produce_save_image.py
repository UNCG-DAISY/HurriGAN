import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import os

def produce_save_image(model_path,city,img_path, seed = None):
    '''
    Generate a new fake image and plot the image.
    Parameters
    ------------------- 
    model_path (str):
        Path of h5 model 
    city (str):
        The city name of the image to generate
    img_path (str):
        Folder path to save image. For current path, input './'
    seed (int):
        Seed of the random noise
        Default None
    '''
    # test input type
    if isinstance(model_path,str) == False:
        raise TypeError("Type of model_path should be string")
    if isinstance(city,str) == False:
        raise TypeError("Type of city should be string")
    if isinstance(img_path,str) == False:
        raise TypeError("Type of img_path should be string")
    if (isinstance(seed,int) == False) & (seed != None):
        raise TypeError("Type of seed should be integer or None")
    # test file path
    if os.path.exists(model_path) == False:
        raise FileNotFoundError("Path of model does not exists")
    if os.path.exists(img_path) == False:
        raise FileNotFoundError("The image folder does not exists") 
    
    
    #label encoded
    cities = ['boston', 'charleston', 'chicago', 'denver', 'houston', 'losangeles', 'miami', 'morgantown', 'portland', 'seattle']
    
    if city.replace(' ','').lower() not in cities:
        raise Exception('Please enter one of the 10 cities')
    
    labels_enc = LabelEncoder().fit(cities)
    label = labels_enc.transform([city.replace(' ','').lower()])
    #load model
    generator = load_model(model_path, compile = False)
    # generate image from random noise
    noise = tf.random.normal([1, 100], seed = seed)
    generated_image = generator([noise,np.array([label])], training=False)
    img_array = generated_image[0, :, :].numpy()
    img_array = (img_array - np.min(img_array))/np.ptp(img_array)
    # plot and save images
    plt.imshow(img_array)
    plt.axis('off')
    plt.savefig(img_path + '/image_{}.png'.format(city.title()))
    print('Generate image for {}'.format(city.title()))
    plt.show()
    plt.close()
