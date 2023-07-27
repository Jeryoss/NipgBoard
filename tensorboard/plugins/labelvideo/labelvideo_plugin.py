from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import glob

import tensorflow as tf
import glob
from werkzeug import wrappers

from tensorboard.backend.http_util import Respond
from tensorboard.plugins import base_plugin
from tensorboard import program

_PLUGIN_PREFIX_ROUTE = 'labelvideo'
_PLUGIN_NAME = 'org_tensorflow_tensorboard_video'
_PLUGINS_DIR = 'plugins'

RUNS_ROUTE = '/runs'
PORT_ROUTE = '/port'
FILE_ROUTE = '/file'
SAVE_ROUTE = '/save'
VIDEOS_ROUTE = '/videos'

class LabelvideoPlugin(base_plugin.TBPlugin):
    plugin_name = _PLUGIN_PREFIX_ROUTE

    def __init__(self, context):
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
        self._handlers = {
            RUNS_ROUTE: self._serve_runs,
            PORT_ROUTE: self._serve_port,
            FILE_ROUTE: self._serve_file_by_name,
            SAVE_ROUTE: self._save_labels,
            VIDEOS_ROUTE: self._load_videos,
        }
        return self._handlers
    
    def is_active(self):
        return True

    def _is_embedding(self,subfolder):
        embedding_folder = "denrun_video__video_faces"
        if subfolder == "":
            print(os.path.join(self.logdir,embedding_folder,"metadata.tsv"))
            print(os.path.exists(os.path.join(self.logdir,embedding_folder,"metadata.tsv")))
            return os.path.exists(os.path.join(self.logdir,embedding_folder,"metadata.tsv"))
        else:
            print(os.path.join(self.logdir,subfolder,embedding_folder,"metadata.tsv"))
            print(os.path.exists(os.path.join(self.logdir,subfolder,embedding_folder,"metadata.tsv")))
            return os.path.exists(os.path.join(self.logdir,subfolder,embedding_folder,"metadata.tsv"))            

    def _determine_is_active(self):
        if self.configs:
            self._is_active = True
        self._thread_for_determining_is_active = None

    @property
    def configs(self):
        return self._configs

    def _run_paths_changed(self):
        num_run_paths = len(list(self.run_paths.keys()))
        if num_run_paths != self.old_num_run_paths:
            self.old_num_run_paths = num_run_paths
            return True
        return False

    def _get_reader_for_run(self, run):
        if run in self.readers:
            return self.readers[run]

        config = self._configs[run]
        reader = None
        if config.model_checkpoint_path:
            try:
                reader = tf.pywrap_tensorflow.NewCheckpointReader(
                    config.model_checkpoint_path)
            except Exception:
                tf.logging.warning('Failed reading "%s"',
                                   config.model_checkpoint_path)
        self.readers[run] = reader
        return reader

    @wrappers.Request.application
    def _serve_runs(self, request):
        keys = ['.']
        return Respond(request, list(keys), 'application/json')
    
    @wrappers.Request.application
    def _serve_port(self, request):
        return Respond(request, str(program.global_port), "text/plain", 200)
    
    @wrappers.Request.application
    def _serve_file_by_name(self, request):
        subfolder = request.args.get("subfolder")
        if(subfolder==''):
            try:
                with tf.gfile.GFile(os.path.join(self.logdir, request.args.get('folder'), request.args.get('fname')), 'r') as json_file:
                    return Respond(request, json.dumps(json.load(json_file)), 'application/json', 200)
            except:
                #print("side i")
                return Respond(request, "Overlay not found", 'text/plain', 404)
        else:
            try:
                with tf.gfile.GFile(os.path.join(self.logdir, subfolder, request.args.get('folder'), request.args.get('fname')), 'r') as json_file:
                    return Respond(request, json.dumps(json.load(json_file)), 'application/json', 200)
            except:
                #print("side ii")
                return Respond(request, "Overlay not found", 'text/plain', 404)
    
    @wrappers.Request.application
    def _save_labels(self, request):
        if sys.version_info[0] < 3:
            labels = request.get_data()
        else:
            labels = request.get_data().decode("utf-8")
        print("LABELS ARE:")
        print(labels)
        subfolder = request.args.get("subfolder")
        if(subfolder==''):
            with tf.gfile.GFile(os.path.join(self.logdir, request.args.get('folder'), request.args.get('fname')), 'w') as json_file:
                json_file.write(labels)
                return Respond(request, "OK", 'text/plain', 200)
        else:
            with tf.gfile.GFile(os.path.join(self.logdir, subfolder, request.args.get('folder'), request.args.get('fname')), 'w') as json_file:
                json_file.write(labels)
                return Respond(request, "OK", 'text/plain', 200)
    
    @wrappers.Request.application
    def _load_videos(self, request):
        cnf_file = "cnf.json"
        subfolder = request.args.get('subfolder')
        subfolder_path = ""
        if(subfolder==''):
            subfolder_path = os.path.join(self.logdir, cnf_file)
        else:
            subfolder_path = os.path.join(self.logdir, subfolder, cnf_file)
        with tf.gfile.GFile(subfolder_path, 'r') as cnf_file:
            config = json.load(cnf_file)
            data = {}
            video_folder = config["default"]["video_folder"]
            video_fps = config["default"]["video_fps"]
            labels = config["default"]["labels"]
            video_res = config["default"]["video_res"]
            if(subfolder==''):
                os.chdir(os.path.join(self.logdir, video_folder))
            else:
                os.chdir(os.path.join(self.logdir, subfolder, video_folder))
            videos = glob.glob("*.mp4")
            data["videos"] = sorted(videos)
            data["fps"] = video_fps
            data["video_folder"] = video_folder
            data["labels"] = labels
            data["video_res"] = video_res
            os.chdir(os.path.join(self.logdir))
            return Respond(request, json.dumps(data), 'application/json', 200)
