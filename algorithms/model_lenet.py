import os
from typing import Optional

from keras.applications.densenet import DenseNet169
from keras.applications.resnet import ResNet101
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications import InceptionV3


def replace_top_layer(model: Model, new_num_classes: int) -> Model:
    """Replaces the top layer of a Keras model with a new layer.

    Args:
        model (keras.models.Model): The Keras model to modify.
        new_num_classes (int): The number of output classes for the new layer.

    Returns:
        keras.models.Model: The modified Keras model.

    """
    # Remove the original top layer.
    model.layers.pop()
    print('Original model')
    model.summary()

    # Add a new top layer with the desired number of output classes.
    x = model.layers[-1].output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, name="fc1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(new_num_classes)(x)
    x = Activation("softmax")(x)

    final_model = Model(inputs=model.input, outputs=x)

    print('Final model')
    final_model.summary()

    return final_model


class LeNet:
    """
    A class to create different types of deep learning models for image classification.

    Attributes:
        None

    Methods:
        build(width, height, depth, classes, weightsPath=None):
            Creates a LeNet model.

        vgg_build(width, height, depth, classes, weightsPath=None):
            Creates a VGG16-based model.

        dense_build(width, height, depth, classes, weightsPath=None):
            Creates a DenseNet169-based model.

        resnet50_build(width, height, depth, classes, weightsPath=None):
            Creates a ResNet50-based model.

        resnet101_build(width, height, depth, classes, weightsPath=None):
            Creates a ResNet101-based model.

        inception3_build(width, height, depth, classes, weightsPath=None):
            Creates an InceptionV3-based model.
    """

    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        """
        Creates a LeNet model for image classification.

        Args:
            width (int): The width of the input image.
            height (int): The height of the input image.
            depth (int): The number of channels in the input image.
            classes (int): The number of classes to classify the input images into.
            weightsPath (str, optional): The path to the pre-trained weights file. Defaults to None.

        Returns:
            tensorflow.keras.Model: The LeNet model.
        """
        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Conv2D(20, 5, 5, border_mode="same",
                         input_shape=(width, height, depth)))
        model.add(BatchNormalization(epsilon=1e-3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(50, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model

    @staticmethod
    def vgg_build(width: int, height: int, depth: int, classes: int, weightsPath: Optional[str] = None) -> Model:
        """
        Builds and returns a VGG16 model with a specified number of output classes.

        Args:
            width (int): Width of the input image.
            height (int): Height of the input image.
            depth (int): Depth of the input image.
            classes (int): Number of output classes.
            weightsPath (str, optional): Path to a pre-trained VGG16 weights file. Defaults to None.

        Returns:
            Model: The VGG16 model with specified number of output classes.

        Raises:
            ValueError: If the specified weights file does not exist.

        """
        K.clear_session()
        base_model = VGG16(weights='imagenet', include_top=True, input_shape=(height, width, depth))
        if weightsPath is not None:
            if not os.path.exists(weightsPath):
                raise ValueError(f"The specified weights file '{weightsPath}' does not exist.")
            base_model.load_weights(weightsPath)
        print(base_model.summary())
        x = base_model.layers[-2].output
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        final_model = Model(inputs=base_model.input, outputs=x)

        return final_model

    @staticmethod
    def dense_build(width: int, height: int, depth: int, classes: int, weightsPath: Optional[str] = None) -> Model:
        """Builds a DenseNet169 model with the specified input shape, number of classes, and optional pre-trained weights.

        Args:
            width (int): Width of the input image.
            height (int): Height of the input image.
            depth (int): Depth (number of channels) of the input image.
            classes (int): Number of classes for the classification task.
            weightsPath (str, optional): Path to pre-trained weights file. Defaults to None.

        Returns:
            Model: The compiled DenseNet169 model.

        Raises:
            ValueError: If the specified input shape is not compatible with the model.
            FileNotFoundError: If the specified weights file is not found.

        """
        K.clear_session()
        if classes == 1:
            activation_function = "sigmoid"
        else:
            activation_function = "softmax"

        input_shape = (height, width, depth)

        # Ensure that the input shape is compatible with the model
        if input_shape != (224, 224, 3):
            raise ValueError(f"Input shape {input_shape} not compatible with DenseNet169")

        # Load the pre-trained base model
        base_model = DenseNet169(weights='imagenet', include_top=True, input_shape=input_shape)

        # Print the summary of the base model
        base_model.summary()

        # Load the pre-trained weights if specified
        if weightsPath is not None:
            try:
                base_model.load_weights(weightsPath)
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {weightsPath}")

        # Remove the last layer and add a new dense layer for the number of classes
        x = base_model.layers[-2].output
        x = Dense(classes)(x)

        x = Activation(activation_function)(x)

        # Create the final model with the input and output layers
        final_model = Model(inputs=base_model.input, outputs=x)

        return final_model

    @staticmethod
    def resnet50_build(width, height, depth, classes, weightsPath=None):
        """
        Builds a ResNet50 model with the specified input dimensions, number of output classes, and optionally loads
        pre-trained weights from a specified path.

        Args:
            width (int): The input width of the image.
            height (int): The input height of the image.
            depth (int): The input depth of the image.
            classes (int): The number of output classes for the model.
            weightsPath (str, optional): The file path for pre-trained weights to load into the model.

        Returns:
            A ResNet50 model with the specified input dimensions and number of output classes, with optional
            pre-trained weights loaded.
        """
        K.clear_session()
        base_model = ResNet50(weights='imagenet', include_top=True, input_shape=(height, width, depth))
        print('::::BASE_MODEL::::')
        base_model.summary()
        print('::::BASE_MODEL_END::::')

        if weightsPath is not None:
            base_model.load_weights(weightsPath)

        x = base_model.layers[-2].output
        x = Dense(classes)(x)
        x = Activation("softmax")(x)
        final_model = Model(inputs=base_model.input, outputs=x)

        return final_model

    @staticmethod
    def resnet101_build(width, height, depth, classes, weightsPath=None):
        """Builds a ResNet101 model for image classification.

        Args:
            width (int): Width of input images.
            height (int): Height of input images.
            depth (int): Depth (number of channels) of input images.
            classes (int): Number of classes in the classification task.
            weightsPath (str): Path to weights file to initialize the model.

        Returns:
            A Keras model for image classification.
        """
        K.clear_session()
        base_model = ResNet101(weights='imagenet', include_top=True, input_shape=(width, height, depth))
        base_model.summary()

        if weightsPath is not None:
            base_model.load_weights(weightsPath)

        x = base_model.layers[-2].output
        x = Dense(classes)(x)
        x = Activation("softmax")(x)
        final_model = Model(inputs=base_model.input, outputs=x)

        return final_model

    @staticmethod
    def inception3_build(width, height, depth, classes, weightsPath=None, freeze_layers=0, dropout=0.5, l2_reg=0.01):
        """
        Build an InceptionV3 model for image classification.

        Args:
            width (int): Width of the input image.
            height (int): Height of the input image.
            depth (int): Number of color channels of the input image.
            classes (int): Number of output classes.
            weightsPath (str): Path to pre-trained weights file.
            freeze_layers (int): Number of layers to freeze at the beginning of the model.
            dropout (float): Dropout rate to use in the model.
            l2_reg (float): L2 regularization factor to use in the model.

        Returns:
            keras.models.Model: InceptionV3 model for image classification.
        """
        K.clear_session()
        base_model = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(height, width, depth))

        if weightsPath is not None:
            base_model.load_weights(weightsPath)

        if freeze_layers > 0:
            for layer in base_model.layers[:freeze_layers]:
                layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(classes)(x)
        x = Activation('softmax')(x)

        model = Model(inputs=base_model.input, outputs=x)
        return model
