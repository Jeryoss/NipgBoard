import getopt
import glob
import math
import os
import sys
import json

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile
from divergence_loss import kullback_leibler_divergence, jensen_shannon_divergence, ccl_divergence
from keras import models
from keras import optimizers
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.utils import multi_gpu_model, to_categorical
from model_lenet import LeNet
from tensorboard.plugins.executer.executer_plugin import makeProjector
from tensorflow.keras.backend import clear_session, set_session
from tensorflow.keras.backend import get_session


def reset_keras():
    """
    Reset the Keras session with a new TensorFlow session.
    """
    try:
        del classifier  # this is from global space - change this as you need
    except NameError:
        pass

    sess = get_session()
    clear_session()
    sess.close()

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))


def format_image(im):
    """
    Resize and reformat an image to a standard size and color format.
    """
    bgr = im.split()
    if len(bgr) != 3:
        convim = Image.new("RGB", im.size)
        convim.paste(im)
        bgr = convim.split()
    im = Image.merge("RGB", (bgr[2], bgr[1], bgr[0]))
    im = im.resize((224, 224), Image.ANTIALIAS)
    return np.asarray(im)


class Generator(tf.keras.utils.Sequence):
    """
    Class is a dataset wrapper for better training performance.
    """

    def __init__(self, x_set, y_set, for_kl=False, y_kl=False, batch_size=256):
        self.x = x_set
        self.y = y_set
        self.for_kl = for_kl
        self.y_kl = y_kl
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]

        if self.y_kl:
            batch_y = self.y[:, inds][inds, :]
        elif self.for_kl:
            batch_y = (self.y[inds][:, None] == self.y[inds][None, :]).astype('int32')
        else:
            batch_y = np.eye(10)[self.y[inds]]

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def load_data(image_folder: str, dataset: str, logdir: str, pairs: np.ndarray) -> tuple:
    """Load data from a specified dataset.

    Args:
        image_folder: A string specifying the folder containing the images.
        dataset: A string specifying the dataset to load. Possible values are "mnist", "custom", and "cifar10".
        logdir: A string specifying the directory containing the log files.
        pairs: A numpy array of pairs to use for the custom dataset.

    Returns:
        A tuple containing:
        - (X_train, y_train): A tuple of numpy arrays representing the training data and their labels.
        - (X_test, y_test): A tuple of numpy arrays representing the test data and their labels.
        - number_of_classes: An integer representing the number of classes in the dataset.
        - X_train_orig: A numpy array of the original training data.
        - image_files: A list of strings representing the image filenames for the custom dataset.
    """
    X_test, y_test, X_train_orig, image_files = None, None, None, None

    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255
        number_of_classes = 10

    elif dataset == "custom":
        png_files = glob.glob(os.path.join(logdir, image_folder, '*png'))
        jpg_files = glob.glob(os.path.join(logdir, image_folder, '*jpg'))
        sprite = os.path.join(logdir, image_folder, 'sprite.png')
        if sprite in png_files:
            png_files.remove(sprite)
        image_files = sorted(png_files + jpg_files)
        X_train = np.zeros(shape=(len(image_files), 224, 224, 3), dtype=np.uint8)
        for i, file in enumerate(image_files):
            X_train[i] = format_image(Image.open(file))
        X_train = X_train.astype('float32') / 255.

        # # Swap -1 and 0, copy for embedding
        pairs2 = np.where(pairs == 0, 2, pairs)
        pairs3 = np.where(pairs2 == -1, 0, pairs2)
        y_train = np.where(pairs3 == 2, -1, pairs3)
        # y_train = pairs

        #
        # X_train = np.where(X_train == -1, 1, X_train)
        # X_test = np.where(X_test == -1, 1, X_test)

        X_train_orig = X_train.copy()

        # Drop full -1 rows and columns
        keep = np.any(y_train != -1, axis=0)
        assert np.any(keep)
        X_train = X_train[keep]
        y_train = y_train[keep, :][:, keep]
        y_train = np.where(y_train == -1, 0, y_train)
        print(np.unique(y_train))
        # raise Exception(np.unique(y_train))
        # y_train = y_train.astype('float32')
        number_of_classes = len(np.unique(y_train))  # len(image_files)
    else:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255
        number_of_classes = 10
        y_train = to_categorical(y_train, number_of_classes).argmax(axis=-1)
        y_test = to_categorical(y_test, number_of_classes).argmax(axis=-1)
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        # print(np.unqiue(X_train))

    return (X_train, y_train), (X_test, y_test), number_of_classes, X_train_orig, image_files


def extract_features(image_path: str, dataset: str, logdir: str, pairs: int, clustering: int, divergence: str,
                     batch_size: int, gpu: int, max_epoch: int, embedding_path: str, model_name: str, boardPath: str,
                     optimized=False) -> None:
    """Extract features from images using VGG architecture and save them in tensorboard.

    Args:
        image_path (str): The path to the images directory.
        dataset (str): The name of the dataset.
        logdir (str): The directory to save the tensorboard logs.
        pairs (int): The number of image pairs for training.
        clustering (int): Whether to use clustering or not.
        divergence (str): The divergence metric to use.
        batch_size (int): The size of the batches for training.
        gpu (int): The number of GPUs to use.
        max_epoch (int): The maximum number of epochs to train for.
        embedding_path (str): The directory to save the embedding data.

    Returns:
        None
    """
    # Load data
    (x_train, y_train), (x_test, y_test), num_classes, x_train_orig, image_files = load_data(image_path, dataset,
                                                                                             logdir, pairs)
    # Build the model

    if model_name == "VGG":
        model = LeNet.vgg_build(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3],
                                classes=num_classes)
        layer_number = -3
    elif model_name == "DenseNet":
        model = LeNet.dense_build(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3],
                                  classes=num_classes)
        layer_number = -3
    else:
        model = LeNet.vgg_build(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3],
                                classes=num_classes)
        layer_number = -3

    # Freeze all layers except the last 5
    for layer in model.layers[:-5]:
        layer.trainable = False

    # Choose the loss function based on the divergence metric
    if clustering == 1:
        print('clustering')
        if divergence == 'kl':
            print("[INFO] kullback_leibler_divergence...")
            loss = kullback_leibler_divergence(2)
        elif divergence == 'jn':
            print("[INFO] jensen_shannon_divergence...")
            loss = jensen_shannon_divergence(0.5)
        else:
            loss = ccl_divergence()
        generator = Generator(x_train, y_train, True, True, batch_size)
    else:
        print('no clustering')
        divergence = 'no'
        loss = 'categorical_crossentropy'
        generator = Generator(x_train, y_train, False, False, False, batch_size)

    opt = optimizers.Adam()

    if gpu > 1:
        model = multi_gpu_model(model, gpus=gpu)

    model.compile(loss=loss, optimizer=opt)

    # Train the model
    print("[INFO] training...")
    model.fit_generator(generator,
                        epochs=max_epoch,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False,
                        shuffle=True,
                        initial_epoch=0)

    # Get the intermediate output of the "fc2" layer
    intermediate_layer_model = models.Model(inputs=model.input, outputs=model.layers[layer_number].output)
    intermediate_output = intermediate_layer_model.predict(x_train_orig)
    intermediate_output = intermediate_output.reshape(-1, np.prod(intermediate_output.shape[1:]))

    if optimized:
        model_path = os.path.join(logdir, f'model_{model_name.lower()}run_organized_ver_')

        saves = sorted(glob.glob(model_path + '*'))
        if len(saves) == 0:
            id_counter = 1
            id_delete = 0

        else:
            last_id = int(saves[-1][-1])  # change this approach to filter model saves differently
            id_counter = last_id + 1

            how_many_to_save = 3  # change this number to increase saved number of models
            id_delete = 0 if last_id < how_many_to_save else int(saves[0][-1])

        model.save(model_path + str(id_counter))

        if id_delete > 0:
            os.remove(model_path + str(id_delete))

        # intermediate_layer_model = models.Model(inputs=model.input, outputs=model.layers[-3].output)
        # intermediate_output = intermediate_layer_model.predict(x_train_orig)
        # intermediate_output = intermediate_output.reshape(-1, np.prod(intermediate_output.shape[1:]))

        embedding_path = f"{model_name.lower()}run_organized_epoch_{str(max_epoch)}_" + str(id_counter) + "__" + image_path

        with open(os.path.join(logdir, "cnf.json"), "r+") as config:
            cnf = json.load(config)
            config.seek(0)
            newalg = {"algorithm_path": os.path.join(boardPath, 'algorithms'),
                      "embedding_folder": f"{model_name.lower()}run_organized", "type": "base",
                      "train": f"model_{model_name.lower()}run_organized_ver_" + str(id_counter),
                      "name": f"{model_name.lower()} Organized " + str(id_counter),
                      "algorithm": {"callable": "run", "keyword_arguments": {"epoch": max_epoch}, "arguments": [],
                                    "file": "model_custom"}}
            cnf["trainings"].append(newalg)
            json.dump(cnf, config)
            config.truncate()
            config.close()


    write_projections(logdir, intermediate_output, embedding_path, image_files)


def write_projections(logdir, intermediate_output, embedding_path, image_files):
    # Extract the labels from the image filenames
    labels = [os.path.basename(img_path).split('_')[0] for img_path in image_files]

    makeProjector(logdir=logdir,
                  embedding_data=intermediate_output,
                  embedding_name='fc1_embedding',
                  embedding_folder=embedding_path,
                  image_paths=image_files,
                  labels=labels)


def arg_parser(argv, logdir=None, boardPath=None, imagePath=None, embeddingPath=None, pairs=None, epoch=None,
               model_name="VGG", optimized=False):
    """
    Runs the main program.

    Args:
        argv: a list of command-line arguments.
        logdir: (optional) a string indicating the directory to save logs.
        boardPath: (optional) a string indicating the directory to save TensorBoard data.
        imagePath: (optional) a string indicating the directory containing images.
        embeddingPath: (optional) a string indicating the file path to save the embedding.
        pairs: (optional) a list of tuples containing image pairs to compute distances.
        epoch: (optional) an integer indicating the number of epochs to run.

    Returns:
        None
    """

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    help_str = """
    -s, --save                  (default save)      subdirectory to save logs.
    -b, --batchSize             (default 128)       batch size.
    -r, --learningRate          (default 0.1)       learning rate.
    --learningRateDecay         (default 1e-7)      learning rate decay.
    --epoch_step                (default 25)        epoch step.
    --max_epoch                 (default 100)       maximum number of iterations.
    -d, --dataset               (default "mnist")   mnist or cifar10.
    -k, --backend               (default "cudnn")   nn (for cpu only), cunn, cudnn (fastest).
    -c, --clustering            (default 0)         indicates whether to perform clustering.
    -v, --divergence            (default jn)        indicates which divergence function to use.
    -g, --gpu                   (default 1)         indicates whether to use GPU.
    """

    save = "save"
    batchSize = 4
    learningRate = 0.1
    learningRateDecay = 1e-7
    epoch_step = 1
    max_epoch = int(epoch)
    dataset = "custom"
    backend = "cudnn"
    clustering = 1
    divergence = 'kl'
    gpu = 1

    try:
        opts, args = getopt.getopt(argv, "hs:b:r:a:e:m:d:k:c:v:g:", ["save=",
                                                                     "batchSize=",
                                                                     "learningRate=",
                                                                     "learningRateDecay=",
                                                                     "epoch_step=",
                                                                     "max_epoch=",
                                                                     "dataset=",
                                                                     "backend=",
                                                                     "clustering=",
                                                                     "divergence=",
                                                                     "gpu="])

    except getopt.GetoptError:
        print(help_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('demo.py ' + help_str)
            sys.exit()
        elif opt in ("-s", "--save"):
            save = arg
        elif opt in ("-b", "--batchSize"):
            batchSize = int(arg)
        elif opt in ("-r", "--learningRate"):
            learningRate = float(arg)
        elif opt in ("-a", "--learningRateDecay"):
            learningRateDecay = float(arg)
        elif opt in ("-e", "--epoch_step"):
            epoch_step = int(arg)
        elif opt in ("-m", "--max_epoch"):
            max_epoch = int(arg)
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-k", "--backend"):
            backend = arg
        elif opt in ("-c", "--clustering"):
            clustering = int(arg)
        elif opt in ("-v", "--divergence"):
            divergence = arg
        elif opt in ("-g", "--gpu"):
            gpu = int(arg)

    extract_features(imagePath, dataset,
                     logdir, pairs, clustering, divergence,
                     batchSize, gpu, max_epoch, embeddingPath,
                     model_name, boardPath, optimized)
