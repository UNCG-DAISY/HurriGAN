'''Collects the filenames for exterior and landscape images.

Note: This takes a long time to run.

Usage: img_label.py [--CITIES=<CITIES>] [--IMG_TYPES=<IMG_TYPES>]

Options:
--CITIES=<CITIES>   String divided by commas containing the names of the cities to use  [default: Boston,Charleston,Chicago,Denver,Houston,LosAngeles,Miami,Morgantown,Portland,Seattle]
--IMG_TYPES=<IMG_TYPES>   String divided by commas containing the type of images to return [default: exterior,landscape]
'''

from zipfile import ZipFile
import numpy as np
import re
import pickle
from docopt import docopt

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

opt = docopt(__doc__)

def main(cities_str, img_types_str):
    cities = cities_str.split(",")
    img_types = img_types_str.split(",")
    label_cities(cities, img_types)


def label_cities(cities, img_types):
    ft_vgg_model = load_model('../data/models/ft_vgg_model.h5', compile = False)

    with open('../data/models/kmeans_boston.pkl', 'rb') as f:
        kmeans = pickle.load(f)

    for city in cities:
        print('getting clusters for {}'.format(city))
        img_cities = []
        img_filenames = []
        img_labels = []
        
        archive = ZipFile('../data/zipped_images/{}.zip'.format(city), 'r')
        for entry in archive.infolist():
            with archive.open(entry) as file:
                img = load_img(file, target_size = (112, 112), color_mode = 'rgb')
                img = img_to_array(img)
                img = img.astype('float32') / 255
                img = np.expand_dims(img, axis=0)
            
                feature = ft_vgg_model.predict(img)
                pred = kmeans.predict([feature.flatten()])

                if pred == 0:
                    if 'landscape' in img_types:
                        img_cities.append(city)
                        img_filenames.append(file.name)
                        img_labels.append('landscape')
                if pred == 2:
                    if 'exterior' in img_types:
                        img_cities.append(city)
                        img_filenames.append(file.name)
                        img_labels.append('exterior')

        files = {'city' : np.array(img_cities),
                 'filename': np.array(img_filenames),
                 'label': np.array(img_labels)}

        with open('../data/labels/landext_filenames_{}.pkl'.format(city), 'wb') as f: 
            pickle.dump(files, f, pickle.HIGHEST_PROTOCOL)  

    print('All files were saved in data/labels')

if __name__ == "__main__":
    main(opt["--CITIES"], 
         opt["--IMG_TYPES"])
