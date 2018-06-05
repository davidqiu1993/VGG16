#!/usr/bin/python

"""
VGG-16 network experiments.

Refs:
    - https://keras.io/applications/
"""

__author__    = 'David Qiu'
__email__     = 'david.qiu@jpl.nasa.gov'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2018, NASA Jet Propulsion Laboratory.'


import os, sys
import time, datetime
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

import pdb, IPython


def test_features_extraction(img_path):
    model = VGG16(weights='imagenet', include_top=False)

    img = image.load_img(img_path, target_size=(224, 224))
    x1 = image.img_to_array(img)
    x2 = np.expand_dims(x1, axis=0)
    x3 = preprocess_input(x2)

    features = model.predict(x3)

    print(features)


def test_prediction(img_path):
    model = VGG16(weights='imagenet')

    img = image.load_img(img_path, target_size=(224, 224))
    x1 = image.img_to_array(img)
    x2 = np.expand_dims(x1, axis=0)
    x3 = preprocess_input(x2)

    preds = model.predict(x3)

    pred_labels = decode_predictions(preds, top=3)[0]
    for label in pred_labels:
        print(label)


def main():
    img_path = '../sample_data/elephant.jpg'

    print('Test Features Extraction:')
    test_features_extraction(img_path)
    print('')

    print('Test Prediction:')
    test_prediction(img_path)
    print('')

    IPython.embed()


if __name__ == '__main__':
    main()
