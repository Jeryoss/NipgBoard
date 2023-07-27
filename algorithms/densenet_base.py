"""
This is a Python script that contains example code for feature extraction using a pretrained DenseNet model for an initial
step in the creation of a neural network project. The extracted features are then used to create a projector embedding
that can be visualized using the Embedding Projector of the TensorBoard.

The script imports necessary libraries such as Numpy, Scikit-learn, Keras, and TensorFlow, among others.
The script also contains helper functions for formatting images and creating the necessary data for the Embedding Projector.

The main function requires several parameters such as the directory for the log, the directory for the images,
the directory for the embedding data, and the number of epochs.
The script then extracts features from the image files using a pretrained DenseNet model,
generates the necessary data for the Embedding Projector, and saves the data in the specified directory.

The run function simply calls the main function with the specified parameters.
"""

import glob
import os

import numpy as np
from PIL import Image
from keras import models
from keras.applications.densenet import DenseNet169
from tensorboard.plugins.executer.executer_plugin import makeProjector


def formatImage(im):
    """
    Format an image.

    Args:
        im (PIL.Image): The input image.

    Returns:
        numpy.ndarray: The formatted image as a NumPy array.
    """
    bgr = im.split()
    if len(bgr) != 3:
        convim = Image.new("RGB", im.size)
        convim.paste(im)
        bgr = convim.split()
    im = Image.merge("RGB", (bgr[2], bgr[1], bgr[0]))
    im = im.resize((224, 224), Image.ANTIALIAS)
    return np.asarray(im)


def main(logdir, image_folder, embedding_folder):
    """
    Extract features from image data using the DenseNet169 model.

    Args:
        logdir (str): The path to the log directory.
        image_folder (str): The name of the folder containing the image data.
        embedding_folder (str): The name of the folder in which the projector data will be saved.
    """
    png_files = glob.glob(os.path.join(logdir, image_folder, '*png'))
    jpg_files = glob.glob(os.path.join(logdir, image_folder, '*jpg'))
    print("FILES:")
    print(os.path.join(logdir, image_folder, '*png'))
    sprite = os.path.join(logdir, image_folder, 'sprite.png')

    if sprite in png_files:
        png_files.remove(sprite)
    image_files = sorted(png_files + jpg_files)

    model = DenseNet169(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

    for layer in model.layers:
        print(layer.name)

    intermediate_layer_model = models.Model(inputs=model.input,
                                            outputs=model.layers[-2].output)

    X_train = np.zeros(shape=(len(image_files), 224, 224, 3), dtype=np.uint8)
    for i in range(len(image_files)):
        X_train[i] = formatImage(Image.open(image_files[i]))

    X_train = X_train.astype('float32') / 255.

    print(X_train.shape)

    intermediate_output = intermediate_layer_model.predict(X_train)

    labels = [os.path.basename(img_path).split('_')[0] for img_path in image_files]

    model_denrun = os.path.join(logdir, "model_denrun")

    if not os.path.exists(model_denrun):
        model.save(model_denrun)

    makeProjector(logdir=logdir,
                  embedding_data=intermediate_output,
                  embedding_name='fc1_embedding',
                  embedding_folder=embedding_folder,
                  image_paths=image_files,
                  labels=labels)


def run(logdir, boardPath, imagePath, embeddingPath, epoch=10, train=None):
    """
    Runs the main function.

    Args:
        logdir: Path to the directory where the image and embedding folders are located.
        boardPath: Path to the directory where the tensorboard logs will be saved.
        imagePath: Name of the folder containing the images.
        embedding
    """
    main(logdir, imagePath, embeddingPath)


if __name__ == "__main__":
    execute()
