# authors: Andres Pitta, Braden Tam, Florence Wang and Hanying Zhang
# date: 2020-05-25

''''This script takes the images from zipped files, stores it in a numpy array or a data generator and runs conditional Generative
Adversarial Networks conditioned by type (landscape or exterior)

Usage: gans.py [--ROOT_FOLDER=<ROOT_FOLDER>] [--METHOD=<METHOD>] [--IMG_SIZE=<IMG_SIZE>] [--BATCH_SIZE=<BATCH_SIZE>] [--EPOCHS=<EPOCHS>] [--CHECKPOINT=<CHECKPOINT>]

Options:
--ROOT_FOLDER=<ROOT_FOLDER>   Name of the folder in the AWS S3 bucket in which the images are stored [default: data/]
--METHOD=<METHOD>   Loading data method [default: ram]
[--IMG_SIZE=<IMG_SIZE>]    Image size [default: 100]
[--BATCH_SIZE=<BATCH_SIZE>]    Number of batch size [default: 64]
[--EPOCHS=<EPOCHS>]     Number of epochs [default: 300]
--CHECKPOINT=<CHECKPOINT> Whether to restore previous checkpoints [default: False]
'''

# General
import numpy as np
import os
from zipfile import ZipFile
import time
from docopt import docopt
import random

# For modeling the Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# For images 
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread
from IPython import display

# For the labels
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

# For connecting to AWS
from io import BytesIO

from cdcgans import cdcgans

opt = docopt(__doc__)

def main(root_folder, method, img_size, batch_size, epochs,checkpoint):
    """
    Main entry for the data download script for conditining on cities.
    Parameters
    ---------
    root_folder (str):
        Location of the zip file (including it)
    method (str):
        loading data method: in RAM or image generator ('ram' or 'generator')
    img_size (int):
        input image size
        Default 100
    batch size (int):
        number of batch size
        Default 64
    epochs (int):
        number of epochs
        Default 300
    checkpoint (str):
        whether to restore checkpoints: 'true' for True, else False.
    """
    
    # Check / download data file
    print("Step 1 - Loading the data", end='\n')
    
    types = ['exterior','landscape']

    # Setting the parameters of the data
    img_width = int(img_size)
    img_height = int(img_size)
    batch_size = int(batch_size)
    epochs= int(epochs)

    # Loading the data as a generator
    image_in_batches, steps_per_epoch = load_data(root_folder, types, batch_size, 
                                                 img_width, img_height, method)
    
    print("Step 1 - DONE!", end='\n')

    print("Step 2 - Building the GAN", end='\n')
    make_gan(image_in_batches, steps_per_epoch, batch_size, epochs, img_width, img_height, types,checkpoint)
    #return image_in_batches,steps_per_epoch
    print("Step 2 - DONE!", end='\n')

def make_gan(image_in_batches, steps_per_epoch, batch_size, epochs, img_width, img_height, type_list,checkpoint):
    """
    Initialize hyperparameter for GANs conditining on cities
    Parameters
    ---------
    image_in_batches (list, generator):
        Dataset in batches
    steps_per_epoch (int):
        number of steps per epoch (total images / batch size)
    batch_size (int):
        number of batch size
    epochs (int):
        number of epochs
    img_width (int):
        image width
    img_height (int):
        image height
    city_list (list):
        list of cities
    checkpoint (str):
        whether to restore checkpoints: 'true' for True, else False.
    """
    # Initializing the hyperparameter
    print("Step 2.1 - Creating the model")
    gans = cdcgans(img_size = img_width, noise_dim = 100, n_classes = 2,
               lr_g = 0.0002, lr_d = 0.0002, beta1 = 0.5, num_examples_to_generate = 2, 
               epochs = epochs, batch_size = batch_size, dropout_rate = 0.5, 
               restore_checkpoint = checkpoint.lower() == 'true',
               save_ckpt_path = 'results/gan/training_checkpoints', load_ckp_path = 'results/gan/training_checkpoints',
               city_names = type_list)
    
    
    print("Step 2.2 - Training the model")
            
    gans.train(dataset = image_in_batches, steps_per_epoch = steps_per_epoch, labels = type_list, 
            saving_gap = 50,
            save_img = True, img_path = 'results/gan/examples', 
            save_loss_batch = True, save_loss_epoch = True, loss_path = 'results/gan/loss' )
    
    print("Step 2.3 - Saving generator.h5")
    gans.generator.save("results/gan/weights/generator_ext.h5")

    
#### This portion is for loading the data ####

def load_data(root_folder, types, batch_size, img_width, img_height, method):
    '''
    Loads the data using boto3 and saves in RAM or returns it  as a data generator	
    Parameters	
    -------------------	
    root_folder (str):	
        Location of the input folder in the s3 bucket	
    cities (list):	
        list of cities for the pictures	
    batch_size (int):	
        number of images per batch	
    img_width (int):	
        desired image width for load_image	
    img_height (int):	
        desired image width for load_image	
    method (str):	
        loading data method: in RAM or image generator ('ram' or 'generator').	
    Returns	
    ---------------	
    generator (np.array, np.array): 	
      tuple with the images and its labels (images, labels)  	
    steps_per_epoch (int): 	
      number of steps per epoch (to perform in the generator) 
    '''
    # load data to RAM
    if str.lower(method) == 'ram':
      X,y = get_data(root_folder, types, 
                    batch_size, img_width, img_height)

      data_in_batches = get_batches(X, y, batch_size)

      return list(zip(data_in_batches[0],data_in_batches[1])), len(data_in_batches[0])
    # load data as image generator
    else: 
      image_generator = image_data_generator(root_folder, types, 
                                           batch_size, img_width, img_height)

      steps_per_epoch = get_steps(root_folder, batch_size)

      return image_generator, steps_per_epoch



def get_data(root_folder,types, batch_size,
                        img_width, img_height):
    '''	
    store RGB images and type labels in numpy array	
    Parameters	
    -------------------	
    root_folder (str):	
        Location of the input folder in the s3 bucket	
    types (list):	
        list of types for the pictures	
    batch_size (int):	
        number of images per batch	
    	
    img_width (int):	
        desired image width for load_image	
    img_height (int):	
        desired image width for load_image	
    Returns	
    ---------------	
    tuple (np.array, np.array): 	
        tuple with the images and its labels (images, labels)    	
    '''

    # Initialzing the type
    img_list = []
    types_list = []
        
    # Label encoding the type
    le = LabelEncoder()
    le.fit(types)

    for t in types: ### loop through two zip files
        archive = ZipFile(root_folder + t + '.zip')
        n = 0
        for entry in archive.infolist():
            with archive.open(entry) as file:

                img = Image.open(file)
                img = img.resize((img_width, img_height), Image.ANTIALIAS)

                if img.mode == 'RGB':
                    img_list.append(np.asarray(img))
                    types_list.append(le.transform([t]))
                n+=1

                if n % 10000 == 0:
                  print(f"loading {t} image {n} out of {len(archive.infolist())}")
                  #break #sample data
                  
    return (np.array(img_list), np.array(types_list))

# reference: https://github.com/gsurma/image_generator/blob/master/ImageGeneratorDCGAN.ipynb            
def get_batches(data, labels, batch_size):
    """
    breaks data array into batches
    Parameters
    -------------------
    data (numpy array):
        image data array
    labels (numpy array):
        labels data array
    batch_size (int):
        number of batch size
    """
    batches = []
    label_batches = []
    for i in range(int(data.shape[0]//batch_size)):
        batch = data[i * batch_size:(i + 1) * batch_size]
        label_batch = labels[i * batch_size:(i + 1) * batch_size]
        augmented_images = []
        for img in batch:
            image = array_to_img(img)
            if random.choice([True, False]):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            augmented_images.append(np.asarray(image))
        batch = np.asarray(augmented_images)
        normalized_batch = (batch / 127.5) - 1.0
        batches.append(normalized_batch)
        label_batches.append(label_batch)
    return [batches, label_batches]

## load data as image generator ##
def image_data_generator(root_folder, types, batch_size,
                        img_width, img_height):
    '''	
    data generator object to store RGB images	
    Parameters	
    -------------------	
    root_folder (str):	
        Location of the input folder in the s3 bucket	
    types (list):	
        list of types for the pictures	
    batch_size (int):	
        number of images per batch	
    	
    img_width (int):	
        desired image width for load_image	
    img_height (int):	
        desired image width for load_image	
    Returns	
    ---------------	
    tuple (np.array, np.array): 	
        tuple with the images and its labels (images, labels)    	
    '''
    # Initialzing the city
    img_list = []
    type_list = []

    le = LabelEncoder()
    le.fit(np.array(types))
    
    n = 0
    while True:  
      for t in types:
        archive = ZipFile(root_folder + t + '.zip')  
        for entry in archive.infolist():
            with archive.open(entry) as file:

                img = Image.open(file)
                img = img.resize((img_width, img_height), Image.ANTIALIAS)

                if img.mode == 'RGB':
                    img_list.append(np.asarray(img))
                    type_list.append(le.transform([t]))
                    n+=1

                if n >= batch_size:
                  yield (np.array(img_list)/127.5 - 1,np.array(type_list).reshape(-1,1))
                  img_list = []
                  type_list = []

                  n = 0
                
def get_steps(root_folder, batch_size):
    '''
    Calculates the number of files in the root folder in AWS s3
    Parameters
    -------------------
    root_folder (str):
        Location of the input folder in the s3 bucket
    batch_size (int):
      number of images per batch  
    Returns
    ---------------
    steps_per_epoch (int):
      number of steps per epoch (total images / batch size)
    '''
    
    archive_exterior = ZipFile(root_folder + 'exterior' + '.zip')
    archive_landscape = ZipFile(root_folder + 'landscape' + '.zip')

    generator_len = len(archive_exterior.infolist()) + len(archive_landscape.infolist())

    return generator_len//batch_size

if __name__ == "__main__":
    main(opt["--ROOT_FOLDER"],
         opt["--METHOD"],
         opt["--IMG_SIZE"],
         opt["--BATCH_SIZE"],
         opt["--EPOCHS"],
         opt["--CHECKPOINT"])
