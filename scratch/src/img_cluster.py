'''Fine-tunes a VGG-16 pre-trained model and trains a k-means clustering algorithm.

Usage: img_cluster.py [--N_LAYERS=<N_LAYERS>]

Options:
--N_LAYERS=<N_LAYERS>   Number of layers of the VGG-16 pre-trained model to finetune. [default: 7]
'''

import numpy as np
from zipfile import ZipFile
import pickle 
import re
from docopt import docopt

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

opt = docopt(__doc__)

def main(n_layers):
    img_arrays_labels = load_labelled_imgs()

    x_train, x_valid, y_train, y_valid = split_data(img_arrays_labels)

    ft_vgg_model = fine_tune_vgg(x_train, x_valid, y_train, y_valid, n_layers)

    train_kmeans(img_arrays_labels, ft_vgg_model)


def load_labelled_imgs():
    with open('../data/labels/manual_label_boston_filenames.pkl', 'rb') as f:
        img_manlab = pickle.load(f)

    img_arrays = []
    labels = []

    print('Reading in the maually labelled Boston images...', end = '')
    archive = ZipFile('../data/zipped_images/Boston.zip', 'r')
    for i in range(len(img_manlab)):
        r = re.compile('(Boston/)(.*)')
        filename = r.search(img_manlab[i][0])[2]
        with archive.open(filename) as file:
            img = load_img(file, target_size = (112, 112), color_mode = 'rgb')
            img_arrays.append(img_to_array(img))
    
        if img_manlab[i][1] == 'miscellaneous':
            labels.append('interior')
        else:
            labels.append(img_manlab[i][1])
    print('done')
    
    img_arrays_labels = {'img_arrays': img_arrays,
                        'labels': labels}

    with open('../data/labels/manual_label_boston_arrays.pkl', 'wb') as f: 
        pickle.dump(img_arrays_labels, f, pickle.HIGHEST_PROTOCOL)
    
    return img_arrays_labels 


def split_data(img_arrays_labels):
    x_train = np.array(img_arrays_labels['img_arrays'])
    x_train = x_train.astype('float32') / 255

    label_encoder = LabelEncoder()
    y_train = np.array(img_arrays_labels['labels'])
    y_train = label_encoder.fit_transform(y_train)
    y_train = to_categorical(y_train)

    return train_test_split(x_train, y_train, test_size = 0.2, random_state = 1)

def fine_tune_vgg(x_train, x_valid, y_train, y_valid, n_layers = 7):
    image_size = x_train.shape[1]
    image_channels = x_train.shape[3]
    input_shape = (image_size, image_size, image_channels)

    new_input = Input(shape=input_shape)
    n_classes = 4

    ft_vgg_model = Sequential()
    
    # load the vgg model
    vgg = VGG16(include_top=False, input_tensor=new_input)

    # sets the last n_layers to trainable
    for layer in vgg.layers[:-n_layers]:
        layer.trainable = False

    ft_vgg_model.add(vgg)
    ft_vgg_model.add(Flatten())
    ft_vgg_model.add(Dense(1024, activation='relu'))
    ft_vgg_model.add(Dropout(0.5))
    ft_vgg_model.add(Dense(n_classes, activation='softmax'))

    ft_vgg_model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy'])

    print('Fine tuning VGG-16...', end = '')
    ft_vgg_model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=10,
                    shuffle=True,
                    validation_data=(x_valid, y_valid))

    # remove last 3 layers to get vgg embedding
    for _ in range(3):
        ft_vgg_model.pop()
    print('done')

    ft_vgg_model.save('../data/models/ft_vgg_model.h5')
    print('Saved to data/models.')
    
    return ft_vgg_model


def train_kmeans(img_arrays_labels, ft_vgg_model):
    print('Getting the VGG-16 features...', end = '')
    features = ft_vgg_model.predict(np.array(img_arrays_labels['img_arrays']))
    print('done')

    kmeans = KMeans(n_clusters=4)
    print('Training k-means...', end = '')
    kmeans.fit(features)
    print('done')
    with open('../data/models/kmeans_boston.pkl', 'wb') as f: 
        pickle.dump(kmeans, f, pickle.HIGHEST_PROTOCOL)   
    
    print('Saved to data/models.')

if __name__ == "__main__":
    main(int(opt["--N_LAYERS"]))