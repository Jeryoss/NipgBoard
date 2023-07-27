from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorboard import program
from tensorboard.backend.http_util import Respond
from tensorboard.plugins import base_plugin
from werkzeug import wrappers

_PLUGIN_PREFIX_ROUTE = 'image'
RUNS_ROUTE = '/runs'
PORT_ROUTE = '/port'


class ImagePlugin(base_plugin.TBPlugin):
    """A TensorBoard plugin that displays images from TensorFlow models."""
    plugin_name = _PLUGIN_PREFIX_ROUTE

    def __init__(self, context):
        """Instantiates ImagePlugin.

        Args:
          context: A TBContext instance.
        """
        self.multiplexer = context.multiplexer
        self.logdir = context.logdir
        self._handlers = None
        self.readers = {}
        self.run_paths = None
        self._configs = None
        self.old_num_run_paths = None
        self.config_fpaths = None
        self._is_active = False
        self._thread_for_determining_is_active = None
        if self.multiplexer:
            self.run_paths = self.multiplexer.RunPaths()

    def get_plugin_apps(self):
        """Returns a dictionary of routes to handlers provided by this plugin."""
        self._handlers = {
            RUNS_ROUTE: self._serve_runs,
            PORT_ROUTE: self._serve_port,
        }
        return self._handlers

    def is_active(self):
        """Returns True if this plugin is active, otherwise False."""
        return self._is_active

    def _determine_is_active(self):
        """Determines whether this plugin is active and sets the _is_active flag."""
        if self.configs:
            self._is_active = True
        self._thread_for_determining_is_active = None

    @property
    def configs(self):
        """Returns a dictionary of configuration protos for each run."""
        return self._configs

    def _run_paths_changed(self):
        """Returns True if the number of runs has changed, otherwise False."""
        num_run_paths = len(list(self.run_paths.keys()))
        if num_run_paths != self.old_num_run_paths:
            self.old_num_run_paths = num_run_paths
            return True
        return False

    def _get_reader_for_run(self, run):
        """Returns a CheckpointReader for the given run.

        Args:
          run: The name of the run.

        Returns:
          A CheckpointReader instance, or None if the run does not have a
          model_checkpoint_path set in its configuration proto.
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
        """Handles GET requests to the /runs route.

        Returns a list of all run names in the plugin's context.
        """
        keys = ['.']
        return Respond(request, list(keys), 'application/json')

    @wrappers.Request.application
    def _serve_port(self, request):
        """Handles GET requests to the /port route.

        Returns the port number for the TensorBoard instance.
        """
        return Respond(request, str(program.global_port), "text/plain", 200)
