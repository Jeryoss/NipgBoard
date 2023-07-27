# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The Embedding Projector plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import imghdr
import os
import threading
import json
import sys

import numpy as np
from werkzeug import wrappers

from google.protobuf import json_format
from google.protobuf import text_format

from tensorboard.backend.http_util import Respond
from tensorboard.compat import tf
from tensorboard.plugins import base_plugin
from tensorboard.plugins import plugin_utils
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.util import tb_logging
from werkzeug.exceptions import NotFound as NotFoundError, BadRequest as BadRequestError

import networkx as nx

logger = tb_logging.get_logger()

# The prefix of routes provided by this plugin.
_PLUGIN_PREFIX_ROUTE = 'projector'

# FYI - the PROJECTOR_FILENAME is hardcoded in the visualize_embeddings
# method in tf.contrib.tensorboard.plugins.projector module.
# TODO(@dandelionmane): Fix duplication when we find a permanent home for the
# projector module.
PROJECTOR_FILENAME = 'projector_config.pbtxt'
_PLUGIN_NAME = 'org_tensorflow_tensorboard_projector'
_PLUGINS_DIR = 'plugins'

# Number of tensors in the LRU cache.
_TENSOR_CACHE_CAPACITY = 1

# HTTP routes.
CONFIG_ROUTE = '/info'
TENSOR_ROUTE = '/tensor'
METADATA_ROUTE = '/metadata'
RUNS_ROUTE = '/runs'
BOOKMARKS_ROUTE = '/bookmarks'
SPRITE_IMAGE_ROUTE = '/sprite_image'
PAIRS_ROUTE = '/pairs'
LOAD_ROUTE = '/load'
CHECK_ROUTE = '/check'
DELETE_ROUTE = '/deleteAll'
PORT_ROUTE = '/port'
MERGE_ROUTE = '/merge'
OUTLIER_ROUTE = '/outlier'

_IMGHDR_TO_MIMETYPE = {
    'bmp': 'image/bmp',
    'gif': 'image/gif',
    'jpeg': 'image/jpeg',
    'png': 'image/png'
}
_DEFAULT_IMAGE_MIMETYPE = 'application/octet-stream'


class LRUCache:
    """
    A Least Recently Used (LRU) cache used for storing the last used tensor.

    Args:
        size (int): The maximum number of items that the cache can hold.

    Raises:
        ValueError: If size is less than 1.

    Attributes:
        size (int): The maximum number of items that the cache can hold.
        cache (OrderedDict): An ordered dictionary that maps keys to their corresponding values.

    """

    def __init__(self, size: int):
        if size < 1:
            raise ValueError('The cache size must be >= 1')
        self.size = size
        self.cache = collections.OrderedDict()

    def get(self, key):
        """
        Retrieves the value associated with the given key.

        Args:
            key: The key to look up in the cache.

        Returns:
            The value associated with the given key, or None if the key is not in the cache.

        """
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return None

    def set(self, key, value):
        """
        Adds or updates the given key-value pair in the cache.

        Args:
            key: The key to add or update in the cache.
            value: The value associated with the key.

        Raises:
            ValueError: If value is None.

        """
        if value is None:
            raise ValueError('value must be != None')
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.size:
                self.cache.popitem(last=False)
        self.cache[key] = value


class EmbeddingMetadata:
    """
    Metadata container for an embedding.

    The metadata holds different columns with values used for visualization
    (color by, label by) in the "Embeddings" tab in TensorBoard.

    Attributes:
        num_points (int): The number of points in the embedding.
        column_names (list of str): The names of the columns in the metadata.
        name_to_values (dict): A dictionary that maps column names to their corresponding values.
    """

    def __init__(self, num_points: int):
        """
        Constructs a metadata for an embedding of the specified size.

        Args:
            num_points (int): The number of points in the embedding.
        """
        self.num_points = num_points
        self.column_names = []
        self.name_to_values = {}

    def add_column(self, column_name: str, column_values):
        """
        Adds a named column of metadata values.

        Args:
            column_name (str): The name of the column.
            column_values (array-like): A 1D array-like object holding the column values. Must be
                of length `num_points`. The i-th value corresponds to the i-th point.

        Raises:
            ValueError: If `column_values` is not 1D array-like, or of length `num_points`,
                or if the column name is already used.
        """
        # Sanity checks.
        if isinstance(column_values, np.ndarray) and column_values.ndim != 1:
            raise ValueError('"column_values" should be of rank 1, '
                             'but is of rank %d' % column_values.ndim)
        if len(column_values) != self.num_points:
            raise ValueError('"column_values" should be of length %d, but is of '
                             'length %d' % (self.num_points, len(column_values)))
        if column_name in self.name_to_values:
            raise ValueError('The column name "%s" is already used' % column_name)

        self.column_names.append(column_name)
        self.name_to_values[column_name] = column_values


def _read_tensor_tsv_file(fpath):
    """Reads a tensor in TSV format from file and returns it as a NumPy array.

    Args:
        fpath (str): The path to the file to read.

    Returns:
        numpy.ndarray: A NumPy array containing the tensor data.

    Raises:
        ValueError: If the file does not exist, or if the data in the file is not
            in the expected format.

    """
    if not os.path.exists(fpath):
        raise ValueError('File not found: %s' % fpath)

    with tf.io.gfile.GFile(fpath, 'r') as f:
        tensor = []
        for line in f:
            line = line.rstrip('\n')
            if line:
                try:
                    tensor.append(list(map(float, line.split('\t'))))
                except ValueError:
                    raise ValueError('Invalid data format in file: %s' % fpath)

    if not tensor:
        raise ValueError('File contains no data: %s' % fpath)

    return np.array(tensor, dtype='float32')


def _assets_dir_to_logdir(assets_dir):
    """Converts an assets directory path to a log directory path.

    The function assumes that the assets directory is located inside a log
    directory in the format '.../logdir/plugins/plugin_name/assets/', and returns
    the path to the log directory.

    Args:
        assets_dir (str): The path to the assets directory.

    Returns:
        str: The path to the log directory.

    Raises:
        ValueError: If the assets directory is not located inside a log directory.

    """
    sub_path = os.path.sep + 'plugins' + os.path.sep
    if sub_path not in assets_dir:
        return assets_dir

    parent_dir = os.path.abspath(os.path.join(assets_dir, os.pardir))
    if os.path.basename(parent_dir) != 'assets':
        raise ValueError('Invalid assets directory: %s' % assets_dir)

    grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
    return grandparent_dir


def _latest_checkpoints_changed(configs, run_path_pairs):
    """Checks if the latest checkpoint has changed in any of the runs.

    Args:
      configs: A dictionary of run configurations, indexed by run name.
      run_path_pairs: A list of (run_name, assets_dir) pairs.

    Returns:
      True if the latest checkpoint has changed in any of the runs; False otherwise.

    Raises:
      ValueError: If an error occurs while processing the run configurations or finding the latest checkpoint.
    """
    try:
        for run_name, assets_dir in run_path_pairs:
            if run_name not in configs:
                config = ProjectorConfig()
                config_fpath = os.path.join(assets_dir, PROJECTOR_FILENAME)
                if tf.io.gfile.exists(config_fpath):
                    with tf.io.gfile.GFile(config_fpath, 'r') as f:
                        file_content = f.read()
                    text_format.Merge(file_content, config)
            else:
                config = configs[run_name]

            # See if you can find a checkpoint file in the logdir.
            logdir = _assets_dir_to_logdir(assets_dir)
            ckpt_path = _find_latest_checkpoint(logdir)
            if not ckpt_path:
                continue
            if config.model_checkpoint_path != ckpt_path:
                return True
        return False
    except Exception as e:
        raise ValueError(f"An error occurred while checking for the latest checkpoints: {str(e)}")


def _parse_positive_int_param(request, param_name):
    """Parses and asserts a positive (>0) integer query parameter.

    Args:
        request: The Werkzeug Request object.
        param_name (str): Name of the parameter.

    Returns:
        int: The parsed positive integer parameter, or None if not present, or -1 if the parameter is not a positive integer.

    Raises:
        ValueError: If the parameter is present but not a valid positive integer.
    """
    param = request.args.get(param_name)
    if not param:
        return None
    try:
        param = int(param)
        if param <= 0:
            raise ValueError()
        return param
    except ValueError:
        raise ValueError(f"{param_name} parameter must be a positive integer")


def _rel_to_abs_asset_path(fpath, config_fpath):
    """Converts a relative asset file path to an absolute one.

    Args:
        fpath (str): The relative path to the asset file.
        config_fpath (str): The path to the configuration file.

    Returns:
        str: The absolute path to the asset file.

    Raises:
        ValueError: If the input file path is not a string.
        FileNotFoundError: If the asset file cannot be found.
    """
    if not isinstance(fpath, str):
        raise ValueError("fpath must be a string")
    fpath = os.path.expanduser(fpath)
    if not os.path.isabs(fpath):
        abs_path = os.path.join(os.path.dirname(config_fpath), fpath)
    else:
        abs_path = fpath
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    return abs_path


def _using_tf():
    """Checks whether the TensorFlow API is being used.

    Returns:
        bool: True if the TensorFlow API is being used, False otherwise.
    """
    return not tf.__version__.startswith("stub")


class ProjectorPlugin(base_plugin.TBPlugin):
    """Embedding projector.

    This plugin serves embeddings to the TensorBoard web application for
    visualization with the Embedding Projector.
    """

    plugin_name = _PLUGIN_PREFIX_ROUTE

    def __init__(self, context):
        """Instantiates ProjectorPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self.multiplexer = context.multiplexer
        self.logdir = context.logdir
        self._handlers = None
        self.readers = {}
        self.run_paths = None
        self._configs = {}
        self.old_num_run_paths = None
        self.config_fpaths = None
        self.tensor_cache = LRUCache(_TENSOR_CACHE_CAPACITY)
        self.subfolder = ' '

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

        self._handlers = {
            RUNS_ROUTE: self._serve_runs,
            CONFIG_ROUTE: self._serve_config,
            TENSOR_ROUTE: self._serve_tensor,
            METADATA_ROUTE: self._serve_metadata,
            BOOKMARKS_ROUTE: self._serve_bookmarks,
            SPRITE_IMAGE_ROUTE: self._serve_sprite_image,
            PAIRS_ROUTE: self._receive_pairs,
            LOAD_ROUTE: self._load_pairs,
            CHECK_ROUTE: self._serve_check,
            DELETE_ROUTE: self._serve_deleteAll,
            PORT_ROUTE: self._serve_port,
            MERGE_ROUTE: self._serve_merge,
            OUTLIER_ROUTE: self._serve_outlier
        }
        return self._handlers

    def is_active(self):
        """Determines whether this plugin is active.

        This plugin is only active if any run has an embedding.

        Returns:
          Whether any run has embedding data to show in the projector.
        """
        if not self.multiplexer:
            return False

        if self._is_active:
            # We have already determined that the projector plugin should be active.
            # Do not re-compute that. We have no reason to later set this plugin to be
            # inactive.
            return True

        if self._thread_for_determining_is_active:
            # We are currently determining whether the plugin is active. Do not start
            # a separate thread.
            return self._is_active

        # The plugin is currently not active. The frontend might check again later.
        # For now, spin off a separate thread to determine whether the plugin is
        # active.
        new_thread = threading.Thread(
            target=self._determine_is_active,
            name='ProjectorPluginIsActiveThread')
        self._thread_for_determining_is_active = new_thread
        new_thread.start()
        return False

    def frontend_metadata(self):
        """Overrides the frontend metadata to set the element name to `vz-projector-dashboard`
        and disable reloading.

        Returns:
            The updated metadata.
        """
        return super(ProjectorPlugin, self).frontend_metadata()._replace(
            element_name='vz-projector-dashboard',
            disable_reload=True,
        )

    def _determine_is_active(self):
        """Determines whether the plugin is active.

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
        Returns a map of run paths to `ProjectorConfig` protos.

        If there are no summary event files, the projector should still work,
        treating the `logdir` as the model checkpoint directory.

        Raises:
            ValueError: If the `run_paths` parameter is not set.

        Returns:
            A dictionary mapping run paths to `ProjectorConfig` protos.
        """
        if self.run_paths is None:
            raise ValueError('`run_paths` parameter must be set.')

        run_path_pairs = list(self.run_paths.items())
        self._append_plugin_asset_directories(run_path_pairs)

        # If there are no run path pairs, append the logdir as the single pair.
        if not run_path_pairs:
            run_path_pairs.append(('.', self.logdir))

        if self._run_paths_changed() or _latest_checkpoints_changed(self._configs, run_path_pairs):
            self.readers = {}
            self._configs, self.config_fpaths = self._read_latest_config_files(run_path_pairs)
            self._augment_configs_with_checkpoint_info()

        return self._configs

    def _run_paths_changed(self):
        """
        Checks whether the number of run paths has changed since the last call.

        Returns:
            A boolean indicating whether the number of run paths has changed.
        """
        if len(self.run_paths) != self.old_num_run_paths:
            self.old_num_run_paths = len(self.run_paths)
            return True
        return False

    def _augment_configs_with_checkpoint_info(self):
        """
        Augments the configurations with checkpoint information.

        Raises:
            ValueError: If an error occurs while augmenting the configurations.
        """
        try:
            for run, config in self._configs.items():
                for embedding in config.embeddings:
                    # Normalize the name of the embeddings.
                    if embedding.tensor_name.endswith(':0'):
                        embedding.tensor_name = embedding.tensor_name[:-2]
                    # Find the size of embeddings associated with a tensors file.
                    if embedding.tensor_path and not embedding.tensor_shape:
                        fpath = _rel_to_abs_asset_path(embedding.tensor_path, self.config_fpaths[run])
                        tensor = self.tensor_cache.get((run, embedding.tensor_name))
                        if tensor is None:
                            tensor = _read_tensor_tsv_file(fpath)
                            self.tensor_cache.set((run, embedding.tensor_name), tensor)
                        embedding.tensor_shape.extend([len(tensor), len(tensor[0])])

                reader = self._get_reader_for_run(run)
                if not reader:
                    continue
                # Augment the configuration with the tensors in the checkpoint file.
                special_embedding = None
                if config.embeddings and not config.embeddings[0].tensor_name:
                    special_embedding = config.embeddings[0]
                    config.embeddings.remove(special_embedding)
                var_map = reader.get_variable_to_shape_map()
                for tensor_name, tensor_shape in var_map.items():
                    if len(tensor_shape) != 2:
                        continue
                    embedding = self._get_embedding(tensor_name, config)
                    if not embedding:
                        embedding = config.embeddings.add()
                        embedding.tensor_name = tensor_name
                        if special_embedding:
                            embedding.metadata_path = special_embedding.metadata_path
                            embedding.bookmarks_path = special_embedding.bookmarks_path
                    if not embedding.tensor_shape:
                        embedding.tensor_shape.extend(tensor_shape)

            # Remove configs that do not have any valid (2D) tensors.
            runs_to_remove = []
            for run, config in self._configs.items():
                if not config.embeddings:
                    runs_to_remove.append(run)
            for run in runs_to_remove:
                del self._configs[run]
                del self.config_fpaths[run]

        except Exception as e:
            # Handle any exception that might occur and raise it as a ValueError.
            raise ValueError(f"An error occurred while augmenting the configurations. Error message: {str(e)}") from e

    def _read_latest_config_files(self, run_path_pairs):
        """Reads and returns the projector config files in every run directory.

        Args:
            run_path_pairs: A list of tuples containing the run name and its corresponding
                assets directory.

        Returns:
            A tuple of two dictionaries. The first dictionary maps a run name to its
            corresponding `ProjectorConfig` proto. The second dictionary maps a run name
            to the file path of its corresponding `ProjectorConfig` proto.

        Raises:
            IOError: If there is an error reading the `ProjectorConfig` proto file.

        """
        configs = {}
        config_fpaths = {}
        for run_name, assets_dir in run_path_pairs:
            config = ProjectorConfig()
            config_fpath = os.path.join(assets_dir, PROJECTOR_FILENAME)
            if tf.io.gfile.exists(config_fpath):
                try:
                    with tf.io.gfile.GFile(config_fpath, 'r') as f:
                        file_content = f.read()
                    text_format.Merge(file_content, config)
                except IOError as e:
                    logger.error('Error reading file "%s": %s', config_fpath, e)
                    raise
                has_tensor_files = False
                for embedding in config.embeddings:
                    if embedding.tensor_path:
                        if not embedding.tensor_name:
                            embedding.tensor_name = os.path.basename(embedding.tensor_path)
                        has_tensor_files = True
                        break

                if not config.model_checkpoint_path:
                    # See if you can find a checkpoint file in the logdir.
                    logdir = _assets_dir_to_logdir(assets_dir)
                    ckpt_path = _find_latest_checkpoint(logdir)
                    if not ckpt_path and not has_tensor_files:
                        continue
                    if ckpt_path:
                        config.model_checkpoint_path = ckpt_path

                # Sanity check for the checkpoint file existing.
                if (config.model_checkpoint_path and _using_tf() and
                        not tf.io.gfile.glob(config.model_checkpoint_path + '*')):
                    logger.warn('Checkpoint file "%s" not found',
                                config.model_checkpoint_path)
                    continue
                configs[run_name] = config
                config_fpaths[run_name] = config_fpath
        return configs, config_fpaths

    def _get_reader_for_run(self, run):
        """Returns a checkpoint reader for the given run.

        Args:
            run: A string representing the name of the run.

        Returns:
            A `tf.train.load_checkpoint` reader object if the checkpoint path exists in the
            `ProjectorConfig` of the given run, otherwise `None`.
        """
        if run in self.readers:
            return self.readers[run]

        config = self._configs[run]
        reader = None
        if config.model_checkpoint_path and _using_tf():
            try:
                reader = tf.train.load_checkpoint(config.model_checkpoint_path)
            except Exception:  # pylint: disable=broad-except
                logger.warn('Failed reading "%s"', config.model_checkpoint_path)
        self.readers[run] = reader
        return reader

    def _get_metadata_file_for_tensor(self, tensor_name, config):
        """Returns the metadata file path for the given tensor name and `ProjectorConfig`.

        Args:
            tensor_name: A string representing the name of the tensor.
            config: A `ProjectorConfig` object containing the embeddings information.

        Returns:
            A string representing the absolute path of the metadata file if it exists for the
            given tensor name, otherwise `None`.
        """
        embedding_info = self._get_embedding(tensor_name, config)
        if embedding_info:
            return embedding_info.metadata_path
        return None

    def _get_bookmarks_file_for_tensor(self, tensor_name, config):
        """Returns the bookmarks file path for the given tensor name and `ProjectorConfig`.

        Args:
            tensor_name: A string representing the name of the tensor.
            config: A `ProjectorConfig` object containing the embeddings information.

        Returns:
            A string representing the absolute path of the bookmarks file if it exists for the
            given tensor name, otherwise `None`.
        """
        embedding_info = self._get_embedding(tensor_name, config)
        if embedding_info:
            return embedding_info.bookmarks_path
        return None

    def _canonical_tensor_name(self, tensor_name):
        """Returns the canonical tensor name for the given tensor name.

        Args:
            tensor_name: A string representing the name of the tensor.

        Returns:
            A string representing the canonical name of the tensor in the format "name:index".
        """
        if ':' not in tensor_name:
            return tensor_name + ':0'
        else:
            return tensor_name

    def _get_embedding(self, tensor_name, config):
        """Returns the embedding information for the given tensor name and `ProjectorConfig`.

        Args:
            tensor_name: A string representing the name of the tensor.
            config: A `ProjectorConfig` object containing the embeddings information.

        Returns:
            A `EmbeddingInfo` object if the given tensor name exists in the `ProjectorConfig`,
            otherwise `None`.
        """
        if not config.embeddings:
            return None
        for info in config.embeddings:
            if (self._canonical_tensor_name(info.tensor_name) ==
                    self._canonical_tensor_name(tensor_name)):
                return info
        return None

    def _append_plugin_asset_directories(self, run_path_pairs):
        """Appends the plugin asset directories to the given list of run and asset directory pairs.

        Args:
            run_path_pairs: A list of tuples where the first element is a string representing the
            name of the run and the second element is a string representing the path to the asset
            directory.

        Returns:
            None. The `run_path_pairs` list is modified in-place.
        """
        for run, assets in self.multiplexer.PluginAssets(_PLUGIN_NAME).items():
            if PROJECTOR_FILENAME not in assets:
                continue
            assets_dir = os.path.join(self.run_paths[run], _PLUGINS_DIR, _PLUGIN_NAME)
            assets_path_pair = (run, os.path.abspath(assets_dir))
            run_path_pairs.append(assets_path_pair)

    def edge_to_remove(graph):
        """Finds the edge with highest betweenness centrality score and returns it."""
        centrality_scores = nx.edge_betweenness_centrality(graph)
        return max(centrality_scores, key=centrality_scores.get)

    @wrappers.Request.application
    def _serve_runs(self, request):
        """Returns a list of runs that have embeddings.

        Args:
            request: A Flask request object.

        Returns:
            A Flask response object containing a list of runs that have embeddings in JSON format.
        """
        return Respond(request, list(self.configs.keys()), 'application/json')

    @wrappers.Request.application
    def _serve_config(self, request):
        """Returns the projector configuration for the specified run.

        Args:
            request: A Flask request object with a 'run' parameter.

        Returns:
            A Flask response object containing the projector configuration for the specified run in
            JSON format, or an error message if the specified run does not exist.
        """
        run = request.args.get('run')
        if run is None:
            return Respond(request, 'query parameter "run" is required', 'text/plain', 400)
        if run not in self.configs:
            return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)

        config = self.configs[run]
        return Respond(request,
                       json_format.MessageToJson(config), 'application/json')

    @wrappers.Request.application
    def _serve_metadata(self, request):
        """Returns the metadata file content for a tensor and run.

        Args:
            request (flask.Request): The incoming HTTP request.

        Returns:
            A flask.Response containing the metadata file content for a tensor and run.

        Raises:
            ValueError: If the run or name query parameters are missing or invalid.
            ValueError: If the metadata file does not exist or is not a file.
        """

        run = request.args.get('run')
        if run is None:
            return Respond(request, 'query parameter "run" is required', 'text/plain',
                           400)

        name = request.args.get('name')
        if name is None:
            return Respond(request, 'query parameter "name" is required',
                           'text/plain', 400)

        num_rows = _parse_positive_int_param(request, 'num_rows')
        if num_rows == -1:
            return Respond(request, 'query parameter num_rows must be integer > 0',
                           'text/plain', 400)

        if run not in self.configs:
            return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)

        config = self.configs[run]
        fpath = self._get_metadata_file_for_tensor(name, config)
        if not fpath:
            return Respond(
                request,
                'No metadata file found for tensor "%s" in the config file "%s"' %
                (name, self.config_fpaths[run]), 'text/plain', 400)
        fpath = _rel_to_abs_asset_path(fpath, self.config_fpaths[run])
        if not tf.io.gfile.exists(fpath) or tf.io.gfile.isdir(fpath):
            return Respond(request, '"%s" not found, or is not a file' % fpath,
                           'text/plain', 400)

        num_header_rows = 0
        with tf.io.gfile.GFile(fpath, 'r') as f:
            lines = []
            # Stream reading the file with early break in case the file doesn't fit in
            # memory.
            for line in f:
                lines.append(line)
                if len(lines) == 1 and '\t' in lines[0]:
                    num_header_rows = 1
                if num_rows and len(lines) >= num_rows + num_header_rows:
                    break
        return Respond(request, ''.join(lines), 'text/plain')

    @wrappers.Request.application
    def _serve_tensor(self, request):
        """Returns a tensor in binary format for a given run and name.

        Args:
            request (flask.Request): The HTTP request object.

        Returns:
            A flask.Response object containing the binary data of the tensor.

        Raises:
            Respond: If any required query parameters are missing or invalid, or if the
            requested tensor or tensor file does not exist.
        """

        run = request.args.get('run')
        if run is None:
            return Respond(request, 'query parameter "run" is required', 'text/plain',
                           400)

        name = request.args.get('name')
        if name is None:
            return Respond(request, 'query parameter "name" is required',
                           'text/plain', 400)

        num_rows = _parse_positive_int_param(request, 'num_rows')
        if num_rows == -1:
            return Respond(request, 'query parameter num_rows must be integer > 0',
                           'text/plain', 400)

        if run not in self.configs:
            return Respond(request, 'Unknown run: "%s"' % run, 'text/plain', 400)

        config = self.configs[run]

        tensor = self.tensor_cache.get((run, name))
        if tensor is None:
            # See if there is a tensor file in the config.
            embedding = self._get_embedding(name, config)

            if embedding and embedding.tensor_path:
                fpath = _rel_to_abs_asset_path(embedding.tensor_path,
                                               self.config_fpaths[run])
                if not tf.io.gfile.exists(fpath):
                    return Respond(request,
                                   'Tensor file "%s" does not exist' % fpath,
                                   'text/plain', 400)
                tensor = _read_tensor_tsv_file(fpath)
            else:
                reader = self._get_reader_for_run(run)
                if not reader or not reader.has_tensor(name):
                    return Respond(request,
                                   'Tensor "%s" not found in checkpoint dir "%s"' %
                                   (name, config.model_checkpoint_path), 'text/plain',
                                   400)
                try:
                    tensor = reader.get_tensor(name)
                except tf.errors.InvalidArgumentError as e:
                    return Respond(request, str(e), 'text/plain', 400)

            self.tensor_cache.set((run, name), tensor)

        if num_rows:
            tensor = tensor[:num_rows]
        if tensor.dtype != 'float32':
            tensor = tensor.astype(dtype='float32', copy=False)
        data_bytes = tensor.tobytes()
        return Respond(request, data_bytes, 'application/octet-stream')

    @wrappers.Request.application
    def _serve_bookmarks(self, request):
        """Serves the bookmarks for a tensor in the specified run.

        Args:
            request: The HTTP request object.

        Returns:
            A HTTP response object containing the bookmarks data in JSON format.

        Raises:
            ValueError: If the 'run' query parameter is not provided.
            ValueError: If the 'name' query parameter is not provided.
            ValueError: If the specified run is unknown.
            ValueError: If no bookmarks file is found for the specified tensor.
            IOError: If the bookmarks file is not found or is not a file.
        """
        run = request.args.get('run')
        if not run:
            raise ValueError('query parameter "run" is required')

        name = request.args.get('name')
        if name is None:
            raise ValueError('query parameter "name" is required')

        if run not in self.configs:
            raise ValueError('Unknown run: "%s"' % run)

        config = self.configs[run]
        fpath = self._get_bookmarks_file_for_tensor(name, config)
        if not fpath:
            raise ValueError('No bookmarks file found for tensor "%s" in the config file "%s"' %
                             (name, self.config_fpaths[run]))
        fpath = _rel_to_abs_asset_path(fpath, self.config_fpaths[run])
        if not tf.io.gfile.exists(fpath) or tf.io.gfile.isdir(fpath):
            raise IOError('"%s" not found, or is not a file' % fpath)

        bookmarks_json = None
        with tf.io.gfile.GFile(fpath, 'rb') as f:
            bookmarks_json = f.read()
        return Respond(request, bookmarks_json, 'application/json')

    @wrappers.Request.application
    def _serve_sprite_image(self, request):
        """
        Serves the sprite image for a given tensor.

        Args:
            request: The request object.

        Returns:
            The HTTP response object containing the sprite image.

        Raises:
            BadRequestError: If the required query parameters are missing or invalid.
            NotFoundError: If the sprite image file is not found.
            InternalServerError: If an unexpected error occurs.
        """
        try:
            # Get query parameters
            run = request.args.get('run')
            name = request.args.get('name')

            # Validate query parameters
            if not run:
                raise BadRequestError('Query parameter "run" is required')
            if not name:
                raise BadRequestError('Query parameter "name" is required')
            if run not in self.configs:
                raise BadRequestError('Unknown run: "%s"' % run)

            # Get embedding info
            config = self.configs[run]
            embedding_info = self._get_embedding(name, config)

            # Check if sprite image file exists
            if not embedding_info or not embedding_info.sprite.image_path:
                raise NotFoundError('No sprite image file found for tensor "%s" in the config file "%s"' %
                                    (name, self.config_fpaths[run]))
            fpath = os.path.expanduser(embedding_info.sprite.image_path)
            fpath = _rel_to_abs_asset_path(fpath, self.config_fpaths[run])
            if not tf.io.gfile.exists(fpath) or tf.io.gfile.isdir(fpath):
                raise NotFoundError('"%s" does not exist or is a directory' % fpath)

            # Serve sprite image
            with tf.io.gfile.GFile(fpath, 'rb') as f:
                encoded_image_string = f.read()
            image_type = imghdr.what(None, encoded_image_string)
            mime_type = _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)
            return Respond(request, encoded_image_string, mime_type)

        except BadRequestError as e:
            return Respond(request, str(e), 'text/plain', 400)

        except NotFoundError as e:
            return Respond(request, str(e), 'text/plain', 404)

        except Exception as e:
            tf.logging.error(str(e))
            return Respond(request, 'Internal server error', 'text/plain', 500)

    @wrappers.Request.application
    def _receive_pairs(self, request):
        """
        Receive pairs and update an output data matrix.

        Args:
            request: The HTTP request.

        Returns:
            A Respond object with the updated output data matrix as a string and a 200 status code.

        Raises:
            ValueError: If any of the required parameters are missing or invalid.
            IOError: If the output data file cannot be accessed.

        """
        pairs_n = int(request.args.get('n', default=0))
        subfolder = request.args.get('subfolder', default='')
        embedding_folder = request.args.get('selectedRun', default='')
        self.subfolder = subfolder

        try:
            if subfolder == '':
                output_data = np.load(os.path.join(self.logdir, embedding_folder, 'kiraPW.npy'))
            else:
                output_data = np.load(os.path.join(self.logdir, subfolder, embedding_folder, 'kiraPW.npy'))
        except (FileNotFoundError, IOError):
            output_data = np.full((pairs_n, pairs_n), 0)

        if sys.version_info[0] < 3:
            pairs = request.get_data()
        else:
            pairs = request.get_data().decode("utf-8")
        tf.logging.warning(pairs)

        try:
            json_data = json.loads(pairs)
            first = json_data["first"]
            second = json_data["second"]
            val = json_data["val"]
        except (json.JSONDecodeError, KeyError):
            raise ValueError("Invalid JSON data")

        output_data[first][second] = val
        output_data[second][first] = val

        if subfolder == '':
            output_file = os.path.join(self.logdir, embedding_folder, 'kiraPW.npy')
        else:
            output_file = os.path.join(self.logdir, subfolder, embedding_folder, 'kiraPW.npy')

        try:
            np.save(output_file, output_data)
        except IOError:
            raise IOError("Could not save output data")

        tril = np.tril(output_data, -1)
        pos1 = tril[tril > 0].sum()
        neg1 = tril[tril < 0].sum()
        pos2, neg2 = plugin_utils.countTransitivePairs(output_data)
        transitive = str(pos1) + "&" + str(pos2) + "&" + str(neg1 * -1) + "&" + str(neg2 * -1)
        return Respond(request, transitive, 'text/plain', 200)

    @wrappers.Request.application
    def _load_pairs(self, request):
        """
        Loads the pairs from the kiraPW.npy file for the given `embeddingFolder` and `subfolder`.

        Args:
            request (Request): The incoming HTTP request object.

        Returns:
            Response: The HTTP response object containing the loaded pairs in JSON format.
        """
        try:
            pairs_n = int(request.args.get('n'))
            subfolder = request.args.get('subfolder')
            embeddingFolder = request.args.get('selectedRun')
            self.subfolder = subfolder
            if subfolder == ' ':
                file_path = os.path.join(self.logdir, embeddingFolder, 'kiraPW.npy')
            else:
                file_path = os.path.join(self.logdir, subfolder, embeddingFolder, 'kiraPW.npy')

            if not os.path.exists(file_path):
                return Respond(request, "", 'text/plain')

            data = np.load(file_path)
            tril = np.tril(data, -1)
            pos1 = tril[tril > 0].sum()
            neg1 = tril[tril < 0].sum()
            pos2, neg2 = plugin_utils.countTransitivePairs(data)
            pairs = {"pos": [], "neg": [], "pos1": str(pos1), "pos2": str(pos2), "neg1": str(neg1 * -1),
                     "neg2": str(neg2 * -1)}
            for i in range(pairs_n):
                for j in range(i):
                    val = data[i][j]
                    if val == 1:
                        pairs["pos"].append([i, j])
                    if val == -1:
                        pairs["neg"].append([i, j])
            return Respond(request, json.dumps(pairs), 'text/plain')
        except Exception as e:
            print("Error occurred while loading pairs: %s", str(e))
            return Respond(request, "", 'text/plain', 500)

    @wrappers.Request.application
    def _serve_check(self, request):
        return Respond(request, 'OK', 'text/plain', 200)

    @wrappers.Request.application
    def _serve_deleteAll(self, request) -> wrappers.Response:
        """
        Deletes all data associated with a given subfolder and embedding folder.

        Args:
            request (Request): The HTTP request object.
            subfolder (str): The subfolder containing the data to be deleted.
            embeddingFolder (str): The embedding folder containing the data to be deleted.

        Returns:
            Response: The HTTP response object.

        Raises:
            ValueError: If the subfolder or embeddingFolder parameters are missing.
        """
        subfolder = request.args.get('subfolder')
        embedding_folder = request.args.get('selectedRun')
        if not subfolder or not embedding_folder:
            raise ValueError("Both subfolder and embeddingFolder parameters are required.")
        if subfolder == ' ':
            data_path = os.path.join(self.logdir, embedding_folder, 'kiraPW.npy')
        else:
            data_path = os.path.join(self.logdir, subfolder, embedding_folder, 'kiraPW.npy')
        try:
            data = np.load(data_path)
            backup_path = data_path.replace('.npy', 'backup.npy')
            np.save(backup_path, data)
            os.remove(data_path)
        except FileNotFoundError:
            return Respond(request, 'Data not found', 'text/plain', 404)
        return Respond(request, 'Done', 'text/plain', 200)

    @wrappers.Request.application
    def _serve_port(self, request) -> wrappers.Response:
        """
        Returns the global port used by the program.

        Args:
            request (Request): The HTTP request object.

        Returns:
            Response: The HTTP response object.

        """
        port = str(program.global_port)
        return Respond(request, port, "text/plain", 200)

    @wrappers.Request.application
    def _serve_outlier(self, request) -> wrappers.Response:
        """
        Marks a data point as an outlier in a numpy array.

        Args:
            request (Request): The HTTP request object.

        Returns:
            Response: The HTTP response object.

        Raises:
            ValueError: If outlier is not a valid integer.
            IndexError: If outlier is out of range.
        """
        try:
            outlier = int(request.args.get('id'))
        except (TypeError, ValueError):
            raise ValueError("id must be a valid integer")

        data = np.load(os.path.join(self.logdir, self.subfolder, 'trainParams.npy'))
        pairs_n = data.shape[0]

        if outlier >= pairs_n:
            raise IndexError("id must be within the range of trainParams.npy")

        for i in range(pairs_n):
            if data[outlier][i] == 1:
                data[outlier][i] = -1
            if data[i][outlier] == 1:
                data[i][outlier] = -1

        np.save(os.path.join(self.logdir, self.subfolder, 'trainParams.npy'), data)
        return Respond(request, 'OK', 'text/plain', 200)

    @wrappers.Request.application
    def _serve_merge(self, request) -> wrappers.Response:
        """
        Merges two data points in a numpy array.

        Args:
            request (Request): The HTTP request object.

        Returns:
            Response: The HTTP response object.

        Raises:
            ValueError: If either merge1 or merge2 is not a valid integer.
            IndexError: If either merge1 or merge2 is out of range.
        """
        try:
            merge1 = int(request.args.get('id1'))
            merge2 = int(request.args.get('id2'))
        except (TypeError, ValueError):
            raise ValueError("id1 and id2 must be valid integers")

        data = np.load(os.path.join(self.logdir, self.subfolder, 'trainParams.npy'))
        pairs_n = data.shape[0]

        if merge1 >= pairs_n or merge2 >= pairs_n:
            raise IndexError("id1 and id2 must be within the range of trainParams.npy")

        for i in range(pairs_n):
            if data[merge1][i] == 1:
                data[merge2][i] = 1
            if data[merge2][i] == 1:
                data[merge1][i] = 1

        np.save(os.path.join(self.logdir, self.subfolder, 'trainParams.npy'), data)
        return Respond(request, 'OK', 'text/plain', 200)


def _find_latest_checkpoint(dir_path: str) -> str or None:
    """
    Finds the latest TensorFlow checkpoint file in a directory.

    Args:
        dir_path (str): The directory path to search for the checkpoint.

    Returns:
        str or None: The path to the latest checkpoint file, or None if not found.

    Raises:
        ValueError: If dir_path is not a string or is an empty string.
        TypeError: If dir_path is not a directory.

    """
    if not isinstance(dir_path, str) or not dir_path:
        raise ValueError("dir_path must be a non-empty string")

    if not os.path.isdir(dir_path):
        raise TypeError(f"{dir_path} is not a directory")

    try:
        ckpt_path = tf.train.latest_checkpoint(dir_path)
        if not ckpt_path:
            # Check the parent directory.
            ckpt_path = tf.train.latest_checkpoint(os.path.join(dir_path, os.pardir))
        return ckpt_path
    except tf.errors.NotFoundError:
        return None
