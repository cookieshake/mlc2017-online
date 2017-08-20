# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import tensorflow as tf
import utils

import tensorflow.contrib.slim as slim

"""Contains the base class for models."""
class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    raise NotImplementedError()

class LogisticModel(BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      num_classes: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    net = slim.flatten(model_input)
    output = slim.fully_connected(
        net, num_classes, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

from tensorflow.contrib.slim.python.slim.nets import resnet_v2, vgg, overfeat, alexnet

class ResNetModel(BaseModel):
  def create_model(self, model_input, num_classes=10, is_training=True, **unused_params):
    output = resnet_v2.resnet_v2_50(model_input, num_classes=num_classes)[0]
    output = tf.reshape(output, [-1, num_classes])
    output = tf.nn.softmax(output)
    return {"predictions": output}


class VggModel(BaseModel):
  def create_model(self, model_input, num_classes=10, is_training=True, **unused_params):
    model_input = tf.image.resize_images(model_input, [224, 224])
    output = vgg.vgg_16(model_input, num_classes=num_classes, is_training=is_training)[0]
    output = tf.reshape(output, [-1, num_classes])
    output = tf.nn.softmax(output)
    return {"predictions": output}

class OverfeatModel(BaseModel):
  def create_model(self, model_input, num_classes=10, is_training=True, **unused_params):
    model_input = tf.image.resize_images(model_input, [231, 231])
    output = overfeat.overfeat(model_input, num_classes=num_classes, is_training=is_training)[0]
    output = tf.reshape(output, [-1, num_classes])
    output = tf.nn.softmax(output)
    return {"predictions": output}

class AlexNetModel(BaseModel):
  def create_model(self, model_input, num_classes=10, is_training=True, **unused_params):
    model_input = tf.image.resize_images(model_input, [224, 224])
    output = alexnet.alexnet_v2(model_input, num_classes=num_classes, is_training=is_training)[0]
    output = tf.reshape(output, [-1, num_classes])
    output = tf.nn.softmax(output)
    return {"predictions": output}