'''Creates the zip file of only the specified image types.

Usage: img_zip.py [--CITIES=<CITIES>] [--IMG_TYPES=<IMG_TYPES>]

Options:
--CITIES=<CITIES>   String divided by commas containing the names of the cities to use  [default: Boston,Charleston,Chicago,Denver,Houston,LosAngeles,Miami,Morgantown,Portland,Seattle]
--IMG_TYPES=<IMG_TYPES>   String divided by commas containing the type of images to return [default: exterior,landscape]
'''

from zipfile import ZipFile
import numpy as np
import pickle
import os
from docopt import docopt

opt = docopt(__doc__)

def main(cities_str, img_types_str):
    cities = cities_str.split(",")
    img_types = img_types_str.split(",")
    create_zip(cities, img_types)

def create_zip(cities, img_types):
    os.chdir('../data/zipped_images')

    dict_list = []
    dict_combined = {}

    for city in cities:
        with open('../labels/landext_filenames_{}.pkl'.format(city), 'rb') as f:
            dict_list.append(pickle.load(f))

    for d in dict_list:
        for k in d.keys():
            dict_combined[k] = np.concatenate(list(d[k] for d in dict_list))

    for img_type in img_types:   
        print('Creating zip file for {} images...'.format(img_type))

        with ZipFile('{}test.zip'.format(img_type), 'w') as file:
            for i in range(len(dict_combined['label'])):
                if dict_combined['label'][i] == img_type:
                    file.write('../unzipped_images/' + dict_combined['city'][i] + '/' + dict_combined['filename'][i])
        print('done')

if __name__ == "__main__":
    main(opt["--CITIES"], 
         opt["--IMG_TYPES"])

