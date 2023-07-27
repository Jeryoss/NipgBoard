from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import io
import json
import os
import numpy as np
import tensorflow as tf
import unittest

from werkzeug import test as werkzeug_test
from werkzeug import wrappers

from google.protobuf import text_format

from tensorboard.backend import application
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer  # pylint: disable=line-too-long
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.compat import tf as tf_compat
from tensorboard.plugins import base_plugin
from tensorboard.plugins.projector import projector_config_pb2
from tensorboard.plugins.projector import projector_plugin
from tensorboard.plugins.multidash import multidash_plugin
from tensorboard.util import test_util

tf.compat.v1.disable_v2_behavior()

USING_REAL_TF = tf_compat.__version__ != 'stub'


class MultidashAppTest(tf.test.TestCase):

  def __init__(self, *args, **kwargs):
    super(ProjectorAppTest, self).__init__(*args, **kwargs)
    self.logdir = None
    self.plugin = None
    self.server = None

  def setUp(self):
    self.log_dir = self.get_temp_dir()

  def _SetupWSGIApp(self):
    multiplexer = event_multiplexer.EventMultiplexer(
        size_guidance=application.DEFAULT_SIZE_GUIDANCE,
        purge_orphaned_data=True)
    context = base_plugin.TBContext(
        logdir=self.log_dir, multiplexer=multiplexer)
    self.plugin = projector_plugin.ProjectorPlugin(context)
    wsgi_app = application.TensorBoardWSGIApp(
        self.log_dir, [self.plugin], multiplexer, reload_interval=0,
        path_prefix='')
    self.server = werkzeug_test.Client(wsgi_app, wrappers.BaseResponse)

  def _Get(self, path):
    return self.server.get(path)

  def _GetJson(self, path):
    response = self.server.get(path)
    data = response.data
    if response.headers.get('Content-Encoding') == 'gzip':
      data = gzip.GzipFile('', 'rb', 9, io.BytesIO(data)).read()
    return json.loads(data.decode('utf-8'))

  def _GenerateEventsData(self):
    with test_util.FileWriterCache.get(self.log_dir) as fw:
      event = event_pb2.Event(
          wall_time=1,
          step=1,
          summary=summary_pb2.Summary(
              value=[summary_pb2.Summary.Value(tag='s1', simple_value=0)]))
      fw.add_event(event)

  def _GenerateProjectorTestData(self):
    config_path = os.path.join(self.log_dir, 'projector_config.pbtxt')
    config = projector_config_pb2.ProjectorConfig()
    embedding = config.embeddings.add()
    # Add an embedding by its canonical tensor name.
    embedding.tensor_name = 'var1:0'

    with tf.io.gfile.GFile(os.path.join(self.log_dir, 'bookmarks.json'), 'w') as f:
      f.write('{"a": "b"}')
    embedding.bookmarks_path = 'bookmarks.json'

    config_pbtxt = text_format.MessageToString(config)
    with tf.io.gfile.GFile(config_path, 'w') as f:
      f.write(config_pbtxt)

    # Write a checkpoint with some dummy variables.
    with tf.Graph().as_default():
      sess = tf.compat.v1.Session()
      checkpoint_path = os.path.join(self.log_dir, 'model')
      tf.compat.v1.get_variable('var1',
                                initializer=tf.constant(np.full([1, 2], 6.0)))
      tf.compat.v1.get_variable('var2', [10, 10])
      tf.compat.v1.get_variable('var3', [100, 100])
      sess.run(tf.compat.v1.global_variables_initializer())
      saver = tf.compat.v1.train.Saver(write_version=tf.compat.v1.train.SaverDef.V1)
      saver.save(sess, checkpoint_path)


if __name__ == '__main__':
  tf.test.main()