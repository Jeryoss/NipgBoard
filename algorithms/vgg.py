# Example code for feature extraction for the initial "step 0". This creates a semi-relevant
# starting point using the VGG16 model weights. Refer to the executer_plugin module helper
# functions in order to generate the necessary data for the Embedding Projector, from which
# one can select the correlational pairs and initiate training for "step n". Further instructions
# below within the code.

# !! In order for the import directive to function be sure to execute the script within the NIPGBoard
# main directory or export that directory into your pythonpath. !!

# Author: Ervin Tegles

import os
import sys

if len(sys.argv) != 4:
    print("Incorrect amount of arguments provided! Please call the script as: ")
    print("python vgg.py <logdir> <imagePath> <embeddingPath>")
    print(" <logdir>: The base directory containing everything")
    print(" <imagePath>: Folder name within LOGDIR containing the image database")
    print(" <embeddingPath>: Folder name within LOGDIR where feature extraction will be saved")
    sys.exit()

import numpy as np
from sklearn import svm
import scipy
from PIL import Image
import itertools

from keras import models
from keras.applications.vgg16 import VGG16
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import tensorflow as tf
import glob
import h5py
from tensorflow.python.tools import inspect_checkpoint as chkp

# Import necessary modules for helper functions. Make sure that the NIPGBoard
# location is visible, by either running the script within it or adding it to the
# pythonpath.    
sys.path.append(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])

from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins.executer.executer_plugin import imageTensorToSprite, makeProjector

tf.__version__

def formatImage(im):
    bgr = im.split()
    if len(bgr) != 3:
        convim = Image.new("RGB",im.size)
        convim.paste(im)
        bgr = convim.split()
    im = Image.merge("RGB", (bgr[2], bgr[1], bgr[0]))
    im = im.resize((224, 224), Image.ANTIALIAS)
    return np.asarray(im)

def execute():
        
    # Logdir directory which will contain all necessary data for NIPGBoard
    logdir = sys.argv[1]
    # Name of image database folder within the Logdir
    image_folder= sys.argv[2]
    # Name of output "step 0" projector embedding's folder within the logdir
    embedding_folder = sys.argv[3]
    # So the logdir should look something like this:
    #
    # /home/ervin/traffic
    # --------
    #        |   traffic_images
    #        |------------
    #                    | Picture database
    #        |
    #        |   vgg16
    #        |------------
    #                    | Step 0 projector embedding data
    #        |
    #        |   kira
    #        |------------
    #                    | Step n training data generated via NIPGBoard

    # Fature extraction by parsing the PNG and JPG files and running them through a pretrained VGG16 model.
    png_files = glob.glob(os.path.join(logdir, image_folder, '*png'))
    jpg_files = glob.glob(os.path.join(logdir, image_folder, '*jpg'))
    sprite = os.path.join(logdir, image_folder, 'sprite.png')

    if sprite in png_files:
        png_files.remove(sprite)
    image_files = sorted(png_files + jpg_files)
    
    X_train = np.zeros(shape=(len(image_files), 224, 224, 3), dtype=np.uint8)
    for i in range(len(image_files)):
        X_train[i] = formatImage(Image.open(image_files[i]))
    X_train = X_train.astype('float32')

    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

    intermediate_layer_model = models.Model(inputs=model.input,
                                 outputs=model.get_layer('fc1').output)
    intermediate_output = intermediate_layer_model.predict(X_train)

    # This helper function turns the given numpy tensor into the necessary Projector data
    # for NIPGBoard. The parameters are as follows:
    # - data: the numpy tensor
    # - name: the name for the embedding
    # - logdir: the absolute path for the directory containing everything to execute the NIPGBoard on
    # - image_folder: the folder containing the images within the logdir
    # - embedding_folder: the folder name in which the projector data will be saved.
    #
    # Make sure that these parameters match the configuration settings for the NIPGBoard in
    # <logdir>/cnf.json configuration file!
    makeProjector(logdir=logdir,
        embedding_data=intermediate_output,
        embedding_name='fc1_embedding',        
        embedding_folder=embedding_folder,
        image_paths=image_files)

if __name__ == "__main__":
    execute()
