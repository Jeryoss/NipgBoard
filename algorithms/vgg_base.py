"""
This is a Python script that contains example code for feature extraction using a pretrained VGG model for an initial
step in the creation of a neural network project. The extracted features are then used to create a projector embedding
that can be visualized using the Embedding Projector of the TensorBoard.

The script imports necessary libraries such as Numpy, Scikit-learn, Keras, and TensorFlow, among others.
The script also contains helper functions for formatting images and creating the necessary data for the Embedding Projector.

The main function requires several parameters such as the directory for the log, the directory for the images,
the directory for the embedding data, and the number of epochs.
The script then extracts features from the image files using a pretrained VGG model,
generates the necessary data for the Embedding Projector, and saves the data in the specified directory.

The run function simply calls the main function with the specified parameters.
"""

import glob
import os

import numpy as np
from PIL import Image
from keras import models
from keras.applications.vgg16 import VGG16
from tensorboard.plugins.executer.executer_plugin import makeProjector


def format_image(im):
    """
    Formats an image by converting it to RGB, resizing it to 224x224 pixels,
    and returning it as a numpy array.

    Args:
        im: PIL image.

    Returns:
        Numpy array of the formatted image.
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
    Extracts features from images using a pretrained VGG16 model, saves the
    features as an embedding in the given embedding folder, and generates a
    sprite image for the embedding.

    Args:
        logdir: Path to the directory where the image and embedding folders are located.
        image_folder: Name of the folder containing the images.
        embedding_folder: Name of the folder where the embedding will be saved.
    """
    # Find image files.
    png_files = glob.glob(os.path.join(logdir, image_folder, '*png'))
    jpg_files = glob.glob(os.path.join(logdir, image_folder, '*jpg'))
    sprite = os.path.join(logdir, image_folder, 'sprite.png')

    if sprite in png_files:
        png_files.remove(sprite)
    image_files = sorted(png_files + jpg_files)

    # Load VGG16 model.
    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

    # Get intermediate layer model.
    intermediate_layer_model = models.Model(inputs=model.input,
                                            outputs=model.layers[-2].output)

    # Load images and format them.
    X_train = np.zeros(shape=(len(image_files), 224, 224, 3), dtype=np.uint8)
    for i in range(len(image_files)):
        X_train[i] = format_image(Image.open(image_files[i]))

    X_train = X_train.astype('float32') / 255.


    # Extract features.
    intermediate_output = intermediate_layer_model.predict(X_train)
    # print(intermediate_output.shape)
    # import csv
    #
    # def read_csv(filename):
    #     data = []
    #     with open(filename, 'r') as file:
    #         reader = csv.reader(file)
    #         next(reader)  # Skip the header row if present
    #         for row in reader:
    #             data.append(row[1:])
    #     return data

    # # Example usage
    # filename = '/home/joul/board/nipg-board-v3/algorithms/utils/subset_train_embedding_isic_combined.csv'  # Replace with your CSV file's name or path
    # result = read_csv(filename)
    # intermediate_output = np.array(result)
    # print(intermediate_output.shape)
    # print(len(image_files))
    #
    # print("EXTRACTED>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # Get labels.
    labels = [os.path.basename(img_path).split('_')[0] for img_path in image_files]

    # Save model.
    model_vgg = os.path.join(logdir, "model_vggrun_10")
    if not os.path.exists(model_vgg):
        model.save(model_vgg)

    # Create embedding and sprite image for tensorboard.
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
