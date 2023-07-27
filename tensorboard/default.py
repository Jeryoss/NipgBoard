# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Collection of first-party plugins.

This module exists to isolate tensorboard.program from the potentially
heavyweight build dependencies for first-party plugins. This way people
doing custom builds of TensorBoard have the option to only pay for the
dependencies they want.

This module also grants the flexibility to those doing custom builds, to
automatically inherit the centrally-maintained list of standard plugins,
for less repetition.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import pkg_resources
import cryptography

from tensorboard.compat import tf
from tensorboard.plugins import base_plugin
from tensorboard.plugins.executer import executer_plugin
from tensorboard.plugins.multidash import multidash_plugin
from tensorboard.plugins.core import core_plugin
from tensorboard.plugins.projector import projector_plugin
from tensorboard.plugins.image import image_plugin
from tensorboard.plugins.selected import selected_plugin
from tensorboard.plugins.modelmanager import modelmanager_plugin
from tensorboard.plugins.labelvideo import labelvideo_plugin
from tensorboard.plugins.graphcut import graphcut_plugin


logger = logging.getLogger(__name__)

# Ordering matters. The order in which these lines appear determines the
# ordering of tabs in TensorBoard's GUI.
_PLUGINS = [
    core_plugin.CorePluginLoader(),
    projector_plugin.ProjectorPlugin,
    executer_plugin.ExecuterPlugin,
    multidash_plugin.MultidashPlugin,
    image_plugin.ImagePlugin,
    selected_plugin.SelectedPlugin,
    modelmanager_plugin.ModelmanagerPlugin,
    labelvideo_plugin.LabelvideoPlugin,
    graphcut_plugin.GraphcutPlugin,
]

def get_plugins():
  """Returns a list specifying TensorBoard's default first-party plugins.

  Plugins are specified in this list either via a TBLoader instance to load the
  plugin, or the TBPlugin class itself which will be loaded using a BasicLoader.

  This list can be passed to the `tensorboard.program.TensorBoard` API.

  :rtype: list[Union[base_plugin.TBLoader, Type[base_plugin.TBPlugin]]]
  """

  return _PLUGINS[:]


def get_dynamic_plugins():
  """Returns a list specifying TensorBoard's dynamically loaded plugins.

  A dynamic TensorBoard plugin is specified using entry_points [1] and it is
  the robust way to integrate plugins into TensorBoard.

  This list can be passed to the `tensorboard.program.TensorBoard` API.

  Returns:
    list of base_plugin.TBLoader or base_plugin.TBPlugin.

  [1]: https://packaging.python.org/specifications/entry-points/
  """
  return [
      entry_point.load()
      for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins')
  ]
