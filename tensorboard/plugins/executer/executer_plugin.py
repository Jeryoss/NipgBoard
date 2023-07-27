"""The algorithm execution plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
import threading
import traceback

import cv2
import imageio
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorboard.backend.http_util import Respond
from tensorboard.plugins import base_plugin
from tensorboard.plugins import plugin_utils
from tensorboard.plugins.executer import executer
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from werkzeug import wrappers

# from tensorflow.contrib.tensorboard.plugins import projectors as projector
# from tensorboard.plugins import projector

# The prefix of routes provided by this plugin.
_PLUGIN_PREFIX_ROUTE = 'executer'

# HTTP routes.
RUNS_ROUTE = '/runs'
CNF_ROUTE = '/cnf'
EXECUTE_ROUTE = '/execute'
METACHECK_ROUTE = '/metacheck'
DET_ROUTE = '/details'


def visualize_embeddings(summary_writer, config):
    """Stores a config file used by the embedding projector.

    Args:
      summary_writer: A `tf.summary.FileWriter` used for writing events.
      config: A `tf.contrib.tensorboard.plugins.projector.ProjectorConfig`
        proto that holds the configuration for the projector such as paths to
        checkpoint files and metadata files for the embeddings. If
        `config.model_checkpoint_path` is none, it defaults to the
        `logdir` used by the summary_writer.

    Raises:
      ValueError: If the summary writer does not have a logdir.
      TypeError: If the `summary_writer` or `config` arguments are not the
        correct type.
      IOError: If an error occurs while writing the config file.

    Returns:
      None.
    """
    try:
        # # Sanity checks.
        # if not isinstance(summary_writer, tf.summary.FileWriter):
        #     raise TypeError('summary_writer must be a tf.summary.FileWriter')
        #
        # if not isinstance(config, tf.contrib.tensorboard.plugins.projector.ProjectorConfig):
        #     raise TypeError('config must be a tf.contrib.tensorboard.plugins.projector.ProjectorConfig')

        logdir = summary_writer.get_logdir()

        # Sanity checks.
        if logdir is None:
            raise ValueError('Summary writer must have a logdir')

        # Saving the config file in the logdir.
        config_pbtxt = text_format.MessageToString(config)
        path = os.path.join(logdir, 'projector_config.pbtxt')
        with tf.io.gfile.GFile(path, 'w') as f:
            f.write(config_pbtxt)
    except (TypeError, ValueError, IOError) as e:
        print('An error occurred while visualizing embeddings: %s', e)
        raise e


def imageTensorToSprite(data, imagePath):
    """Converts a tensor of images to a sprite image and saves it to disk.

    Args:
        data: A 4D tensor of images in NHWC format.
        imagePath: The path to save the sprite image to.

    Raises:
        ValueError: If `data` is not a 4D tensor.
        OSError: If there was an error writing the sprite image file.

    """
    if len(data.shape) != 4:
        raise ValueError('Expected a 4D tensor, got {} dimensions'.format(len(data.shape)))

    data = np.tile(data[..., np.newaxis], (1, 1, 1, 3)).astype(np.float32)
    data -= np.min(data.reshape((data.shape[0], -1)), axis=1)[:, np.newaxis, np.newaxis, np.newaxis]
    data /= np.max(data.reshape((data.shape[0], -1)), axis=1)[:, np.newaxis, np.newaxis, np.newaxis]

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0)

    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)

    sprite_path = os.path.join(imagePath, 'sprite.png')
    try:
        imageio.imwrite(sprite_path, data)
    except OSError as e:
        raise OSError('Error writing sprite image file: {}'.format(str(e))) from e


def create_metadata_in_folder(image_path):
    """Creates a metadata file in the given image directory.

    Args:
        image_path (str): The path to the directory containing the images.

    Raises:
        ValueError: If the given directory does not exist or is not a directory.
    """
    if not os.path.exists(image_path) or not os.path.isdir(image_path):
        raise ValueError('Invalid image directory: %s' % image_path)

    images = os.listdir(image_path)
    if 'sprite.png' in images:
        images.remove('sprite.png')
    images = sorted(images)

    metadata_file_path = os.path.join(image_path, 'metadata.tsv')
    with open(metadata_file_path, 'w') as metadata_file:
        metadata_file.write('Filename\t_sync_id\n')
        for i, image_name in enumerate(images):
            metadata_file.write('%s\t%d\n' % (image_name, i))
            print('Printing #%d: %s' % (i, image_name))

    print('Created metadata file at: %s' % metadata_file_path)


def create_sprite_image(img_paths, size):
    """
    Creates a sprite image from a list of image paths.

    Args:
        img_paths: A list of strings containing the paths to the images.
        size: An integer representing the size (in pixels) of the images in the sprite.

    Returns:
        A numpy array representing the sprite image.

    Raises:
        ValueError: If `img_paths` is empty or `size` is not a positive integer.
        OSError: If there is an error reading an image.

    """
    if not img_paths:
        raise ValueError("Image path list is empty")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer")

    n_images = len(img_paths)
    img_h = size
    img_w = size
    n_plots = int(np.ceil(np.sqrt(n_images)))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots, 3), dtype=np.uint8)

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < n_images:
                img_path = img_paths[this_filter]
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (size, size))
                    spriteimage[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = img
                except Exception as e:
                    raise OSError(f"Error reading image at path {img_path}: {str(e)}")

    return spriteimage


def makeProjector(logdir: str, embedding_data: np.ndarray, embedding_name: str,
                  embedding_folder: str, image_paths, labels=None) -> None:
    """Creates a TensorBoard projector with the given data.

    Args:
        logdir (str): The directory where the projector data will be stored.
        embedding_data (np.ndarray): The embedding data to be visualized.
        embedding_name (str): The name of the embedding variable.
        embedding_folder (str): The name of the folder to store the projector data.
        image_paths (List[str]): The paths to the images to be displayed in the projector.
        labels (Optional[List[str]]): A list of labels corresponding to each image path.

    Raises:
        ValueError: If the length of `image_paths` does not match the length of `embedding_data`.
        IOError: If the `logdir` directory cannot be created.
        OSError: If there is an error creating the sprite image or metadata file.
    """
    if not os.path.exists(logdir):
        try:
            os.makedirs(logdir)
        except OSError as e:
            raise IOError(f"Failed to create log directory: {e}")

    if len(image_paths) != len(embedding_data):
        raise ValueError("Length of image_paths must match length of embedding_data.")

    if not os.path.exists(os.path.join(logdir, embedding_folder)):
        os.mkdir(os.path.join(logdir, embedding_folder))

    sprite_size = 28
    sprite_path = os.path.join(logdir, embedding_folder, 'sprite.png')
    meta_path = os.path.join(logdir, embedding_folder, 'metadata.tsv')
    checkpoint_path = os.path.join(logdir, embedding_folder, embedding_name + '.ckpt')

    embedding_var = tf.Variable(embedding_data, name=embedding_name)
    sess = tf.Session()
    sess.run(embedding_var.initializer)
    summary_writer = tf.summary.FileWriter(os.path.join(logdir, embedding_folder))
    summary_writer.add_graph(sess.graph)
    config = ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    embedding.metadata_path = meta_path
    embedding.sprite.image_path = sprite_path
    embedding.sprite.single_image_dim.extend([sprite_size, sprite_size])

    visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver([embedding_var])
    saver.save(sess, checkpoint_path, 1)

    #### SPRITE IMAGE
    print('sprite image')
    try:
        if not os.path.isfile(sprite_path):
            sprite_image = create_sprite_image(image_paths, sprite_size)
            cv2.imwrite(sprite_path, sprite_image)
    except Exception as e:
        raise OSError(f"Failed to create sprite image: {e}")

    ##### META FILE

    print('meta file', meta_path)
    try:
        if not os.path.isfile(meta_path):
            print("No Metadata")

            print('create')
            if labels is None:
                with open(meta_path, 'w') as f:
                    f.write("Index\tFilename\t_sync_id\tClusters\n")
                    for index, img_path in enumerate(image_paths):
                        f.write("%d\t%s\t%d\t%s\n" % (index, os.path.basename(img_path), index, "Unassigned"))
            else:
                with open(meta_path, 'w') as f:
                    f.write("Index\tLabel\tFilename\t_sync_id\tClusters\n")

                    for index, (label, img_path) in enumerate(zip(labels, image_paths)):
                        f.write("%d\t%s\t%s\t%d\t%s\n" % (
                            index, label, img_path.split('/')[-1].split('\\')[-1], index, "Unassigned"))
    except Exception as e:
        print("An exception has occurred: ", e)


class AlgorithmNoPairsError(Exception):
    pass


class ExecuterPlugin(base_plugin.TBPlugin):
    """Executer plugin for TensorBoard.

    This plugin offers a web interface for executing code on embeddings. Users
    can select a specific embedding run, then choose an image from the run's
    directory and execute their code on it.

    Attributes:
        plugin_name (str): Plugin name.
        multiplexer (multiplexer_v2.Multiplexer): TensorBoard data multiplexer.
        logdir (str): Path to the directory containing TensorBoard data.
        _handlers (Dict[str, Callable]): Dictionary of endpoint handlers.
        readers (Dict[str, Any]): Dictionary of TensorFlow checkpoint readers.
        run_paths (Dict[str, str]): Dictionary of run paths.
        _configs (Dict[str, RunConfig]): Dictionary of run configurations.
        old_num_run_paths (int): Number of previous run paths.
        config_fpaths (List[str]): List of run configuration file paths.
        cnf_json (Dict[str, Any]): Dictionary of configuration values.
        _is_active (bool): Whether the plugin is active.
        _thread_for_determining_is_active (threading.Thread): Thread for determining whether the plugin is active.
    """

    plugin_name = _PLUGIN_PREFIX_ROUTE

    def __init__(self, context: base_plugin.TBContext):
        """Instantiates Executer via TensorBoard core.

        Args:
            context: A base_plugin.TBContext instance.
        """
        self.multiplexer = context.multiplexer
        self.logdir = context.logdir
        self._handlers = None
        self.readers = {}
        self.run_paths = None
        self._configs = None
        self.old_num_run_paths = None
        self.config_fpaths = None
        self.cnf_json = None

        # Whether the plugin is active (has meaningful data to process and serve).
        # Once the plugin is deemed active, we no longer re-compute the value
        # because doing so is potentially expensive.
        self._is_active = False

        # The running thread that is currently determining whether the plugin is
        # active. If such a thread exists, do not start a duplicate thread.
        self._thread_for_determining_is_active = None

        if self.multiplexer:
            self.run_paths = self.multiplexer.RunPaths()

    def get_plugin_apps(self):
        """Returns a dictionary mapping URL routes to request handler functions.

        Returns:
            A dictionary with URL routes as keys and handler functions as values.
        """
        self._handlers = {
            RUNS_ROUTE: self._serve_runs,
            CNF_ROUTE: self._serve_cnf,
            EXECUTE_ROUTE: self._serve_execute,
            METACHECK_ROUTE: self._serve_metacheck,
            DET_ROUTE: self._serve_details
        }
        return self._handlers

    def is_active(self) -> bool:
        """Determines whether the plugin is active (has meaningful data to process and serve).

        Returns:
            A boolean indicating whether the plugin is active.
        """
        if self._is_active:
            # The plugin has already been determined to be active.
            return True
        elif self._thread_for_determining_is_active:
            # Another thread is already determining whether the plugin is active.
            return False
        else:
            # Start a new thread to determine whether the plugin is active.
            self._thread_for_determining_is_active = threading.Thread(
                target=self._determine_is_active
            )
            self._thread_for_determining_is_active.start()
            return False

    def _determine_is_active(self):
        """
        Determines whether the plugin is active.

        This method is run in a separate thread so that the plugin can offer an
        immediate response to whether it is active and determine whether it should
        be active in a separate thread.
        """
        if self.configs:
            self._is_active = True
        self._thread_for_determining_is_active = None

    @property
    def configs(self):
        """
        Returns the configs.
        """
        return self._configs

    def _run_paths_changed(self):
        """
        Returns whether the run paths have changed since the last time this method
        was called.
        """
        num_run_paths = len(list(self.run_paths.keys()))
        if num_run_paths != self.old_num_run_paths:
            self.old_num_run_paths = num_run_paths
            return True
        return False

    def _get_reader_for_run(self, run):
        """
        Returns a checkpoint reader for a given run.

        Args:
          run: The name of the run.

        Returns:
          A checkpoint reader for the given run.
        """
        if run in self.readers:
            return self.readers[run]

        config = self._configs[run]
        reader = None
        if config.model_checkpoint_path:
            try:
                reader = tf.pywrap_tensorflow.NewCheckpointReader(
                    config.model_checkpoint_path)
            except Exception as e:
                tf.logging.warning('Failed reading "%s": %s',
                                   config.model_checkpoint_path, str(e))
        self.readers[run] = reader
        return reader

    @wrappers.Request.application
    def _serve_runs(self, request):
        """Returns a list of runs that have embeddings.

        Args:
            request: The request object.

        Returns:
            A list of runs that have embeddings in JSON format.
        """
        keys = ['.']
        return Respond(request, list(keys), 'application/json')

    @wrappers.Request.application
    def _serve_cnf(self, request):
        """Returns the configuration file for a given subfolder if it exists.

        Args:
          request: The HTTP request object.

        Returns:
          The configuration file in JSON format, if it exists.

        Raises:
          HTTPException: If the configuration file cannot be found or is invalid.
        """

        f = "cnf.json"
        subfolder = request.args.get('subfolder')

        if subfolder == ' ':
            if os.path.isfile(os.path.join(self.logdir, f)):
                with tf.gfile.GFile(os.path.join(self.logdir, f), 'r') as json_file:
                    try:
                        self.cnf_json = json.load(json_file)
                        return Respond(request, json.dumps(self.cnf_json), 'application/json', 200)
                    except json.JSONDecodeError:
                        return Respond(request, "Invalid config file!", "text/plain", 500)
            else:
                return Respond(request, "Config file not found!", "text/plain", 500)
        else:
            if os.path.isfile(os.path.join(self.logdir, subfolder, f)):
                with tf.gfile.GFile(os.path.join(self.logdir, subfolder, f), 'r') as json_file:
                    try:
                        self.cnf_json = json.load(json_file)
                        return Respond(request, json.dumps(self.cnf_json), 'application/json', 200)
                    except json.JSONDecodeError:
                        return Respond(request, "Invalid config file!", "text/plain", 500)
            else:
                return Respond(request, "Config file not found!", "text/plain", 500)

    @wrappers.Request.application
    def _serve_execute(self, request):

        """Serve the execute request from the client.

           Args:
               request: HTTP request object containing parameters.

           Returns:
               A tuple of (response content, content type, HTTP status code).

           Raises:
               ValueError: If required parameters are missing.
               FileNotFoundError: If file path is not found.
        """
        # Double checks image data, instantiates execution, and handles error.

        boardPath = request.args.get("boardPath")
        imagePath = request.args.get('imagePath')
        num = int(request.args.get('num'))
        subfolder = request.args.get('subfolder')
        embeddingFolder = request.args.get('selectedRun')

        # sys.path.append(self.logdir)
        if subfolder == ' ':
            noImages = len(glob.glob(os.path.abspath(os.path.join(self.logdir, subfolder, imagePath, '*.png'))))
            noImages = noImages + len(
                glob.glob(os.path.abspath(os.path.join(self.logdir, subfolder, imagePath, '*.jpg'))))
            noImages = noImages + len(
                glob.glob(os.path.abspath(os.path.join(self.logdir, subfolder, imagePath, '*.jpeg'))))
            if noImages < 1:
                return Respond(request, "Error: no images found in directory!", "text/plain", 500)
            if not os.path.exists(os.path.abspath(os.path.join(self.logdir, embeddingFolder, 'kiraPW.npy'))):
                return Respond(request, "Error: There are no pairs to train on!", 'text/plain', 500)

            pairs = np.load(os.path.join(self.logdir, embeddingFolder, 'kiraPW.npy'))

            if len(pairs) != noImages:
                return Respond(request,
                               "Error: trying to train on old pairs on different images. Please delete the pairs and make new ones.",
                               "text/plain", 500)
            useful_rows = np.any(pairs != -1, axis=0)
            # if not np.any(useful_rows):
            # return Respond(request, "Error: There are no pairs to train on!", 'text/plain', 500)
            pairs = plugin_utils.pairWise2Transitive(pairs)
            alg = self.cnf_json['trainings'][num]

            ctx = {'logdir': self.logdir, 'boardPath': boardPath, 'imagePath': imagePath,
                   'embeddingPath': alg['embedding_folder'], 'pairs': pairs}

        else:
            noImages = len(glob.glob(os.path.abspath(os.path.join(self.logdir, subfolder, imagePath, '*.png'))))
            noImages = noImages + len(
                glob.glob(os.path.abspath(os.path.join(self.logdir, subfolder, imagePath, '*.jpg'))))
            noImages = noImages + len(
                glob.glob(os.path.abspath(os.path.join(self.logdir, subfolder, imagePath, '*.jpeg'))))
            if noImages < 1:
                return Respond(request, "Error: no images found in directory!", "text/plain", 500)

            trainings = []
            for training in self.cnf_json['trainings']:
                if training["type"] == "train":
                    trainings.append(training)

            alg = trainings[num]

            pairs = []
            if alg["train"] == "paired":
                if not os.path.exists(
                        os.path.abspath(os.path.join(self.logdir, subfolder, embeddingFolder, 'kiraPW.npy'))):
                    return Respond(request, "Error: There are no pairs to train on!", 'text/plain', 500)
                pairs = np.load(os.path.join(self.logdir, subfolder, embeddingFolder, 'kiraPW.npy'))
                useful_rows = np.any(pairs != -1, axis=0)
                if not np.any(useful_rows):
                    return Respond(request, "Error: There are no pairs to train on!", 'text/plain', 500)
                pairs = plugin_utils.pairWise2Transitive(pairs)
                if len(pairs) != noImages:
                    return Respond(request,
                                   "Error: trying to train on old pairs on different images. Please delete the pairs and make new ones.",
                                   "text/plain", 500)

            ctx = {'logdir': self.logdir + '/' + subfolder, 'boardPath': boardPath, 'imagePath': imagePath,
                   'embeddingPath': alg['embedding_folder'] + "__" + imagePath, 'pairs': pairs}

        # Instantiate the class responsible for proper script execution with proper parameter context.
        executable = executer.PythonExecuter(config_list=alg, context_list=ctx)
        # Import directives and other necesseary prerequisites.
        executable.load_statements()
        executable.make_higher_order()
        print("before_try_catch")

        try:
            executable.execute_functions_pool()
            print("try_end")
        except (NameError, SyntaxError, ImportError) as e:
            tf.logging.error("\n" + str(e))
            tf.logging.error(traceback.format_exc())
            return Respond(request, "Error in algorithm's Python script!", 'text/plain', 500)
        except (MemoryError, tf.errors.ResourceExhaustedError) as e:
            tf.logging.error("\n" + str(e))
            tf.logging.error(traceback.format_exc())
            return Respond(request, "Error: CPU or GPU memory is exhausted!", 'text/plain', 500)
        except Exception as e:
            tf.logging.error("\n" + str(e))
            tf.logging.error(traceback.format_exc())
            return Respond(request, "", 'text/plain', 500)

        return Respond(request, "", 'text/plain', 200)

    @wrappers.Request.application
    def _serve_details(self, request):
        """
        Returns algorithm details for a given number and subfolder.

        Args:
            request: A Flask request object containing the number and subfolder.

        Returns:
            A Flask response object with the algorithm details in JSON format.

        Raises:
            None
        """
        num = int(request.args.get('num'))
        subfolder = request.args.get('subfolder')

        dets = {}

        if subfolder == '':
            logdir = self.logdir
            alg = self.cnf_json['trainings'][num]

        else:
            logdir = self.logdir + subfolder
            trainings = []
            for training in self.cnf_json['trainings']:
                if training["type"] == "train":
                    trainings.append(training)

            alg = trainings[num]

        dets["name"] = alg["name"]
        dets["algorithm_path"] = alg["algorithm_path"]
        dets["embedding_folder"] = alg["embedding_folder"]
        dets["file"] = alg["algorithm"]["file"]
        dets["type"] = alg["type"]
        dets["train"] = alg["train"]

        if dets["train"] == "organized":
            model_path = os.path.join(logdir, 'model_' + dets["embedding_folder"].split('_')[0] + '_organized_ver_')
            saves = sorted(glob.glob(model_path + '*'))
            if len(saves) == 0:
                id_counter = 1
            else:
                last_id = int(saves[-1][-1])
                id_counter = last_id + 1
            dets["embedding_folder"] = dets["embedding_folder"] + "_" + str(id_counter)

        return Respond(request, json.dumps(dets), 'application/json')

    def createMetadataInFolder(self, imagePath):
        """Creates a metadata.tsv file in the given directory containing metadata for the images in the directory.

        Args:
          imagePath: A string representing the path to the directory containing images.
        """
        metadataPath = os.path.join(imagePath, 'metadata.tsv')
        with open(metadataPath, 'w') as metadataFile:
            metadataFile.write('Index\tLabel\n')
            for i, filename in enumerate(os.listdir(imagePath)):
                metadataFile.write('{}\t{}\n'.format(i, filename))

    @wrappers.Request.application
    def _serve_metacheck(self, request):
        """Checks for missing metadata.tsv and sprite.png files and generates them if missing.

        Args:
            request: The HTTP request.

        Returns:
            An HTTP response indicating whether the operation succeeded.
        """
        # try:
        #     embedding_folder = self.cnf_json['default']['embedding_folder']
        #     image_path = os.path.join(self.logdir, embedding_folder)
        #     if not os.path.isfile(os.path.join(image_path, 'metadata.tsv')):
        #         self.createMetadataInFolder(image_path)
        #     if not os.path.isfile(os.path.join(image_path, 'sprite.png')):
        #         self.createMetadataInFolder(image_path)
        # except Exception as e:
        #     return Respond(request, f"Error: {str(e)}", 'text/plain', 500)

        return Respond(request, "OK", 'text/plain', 200)
