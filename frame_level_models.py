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
"""Contains a collection of models which operate on variable-length sequences."""
import math

import model_utils as utils
import models
import tensorflow as tf
from tensorflow import flags
import tensorflow.contrib.slim as slim
import video_level_models

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30, "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string(
    "dbof_pooling_method", "max",
    "The pooling method used in the DBoF cluster layer. "
    "Choices are 'average' and 'max'.")
flags.DEFINE_string(
    "dbof_activation", "sigmoid",
    "The nonlinear activation method for cluster and hidden dense layer, e.g., "
    "sigmoid, relu6, etc.")
flags.DEFINE_string(
    "video_level_classifier_model", "MoeModel",
    "Some Frame-Level models can be decomposed into a "
    "generalized pooling operation followed by a "
    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")


class FrameLevelLogisticModel(models.BaseModel):
    """Creates a logistic classifier over the aggregated frame-level features."""

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """See base class.

        This class is intended to be an example for implementors of frame level
        models. If you want to train a model over averaged features it is more
        efficient to average them beforehand rather than on the fly.

        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
            input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
            frames for each video (before padding).

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        feature_size = model_input.get_shape().as_list()[2]

        denominators = tf.reshape(tf.tile(num_frames, [1, feature_size]),
                                  [-1, feature_size])
        avg_pooled = tf.reduce_sum(model_input, axis=[1]) / denominators

        output = slim.fully_connected(avg_pooled,
                                      vocab_size,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_regularizer=slim.l2_regularizer(1e-8))
        return {"predictions": output}


class DbofModel(models.BaseModel):
    """Creates a Deep Bag of Frames model.

    The model projects the features for each frame into a higher dimensional
    'clustering' space, pools across frames in that space, and then
    uses a configurable video-level model to classify the now aggregated features.

    The model will randomly sample either frames or sequences of frames during
    training to speed up convergence.
    """

    ACT_FN_MAP = {
        "sigmoid": tf.nn.sigmoid,
        "relu6": tf.nn.relu6,
    }

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        """See base class.

        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
            input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
            frames for each video (before padding).
          iterations: the number of frames to be sampled.
          add_batch_norm: whether to add batch norm during training.
          sample_random_frames: whether to sample random frames or random sequences.
          cluster_size: the output neuron number of the cluster layer.
          hidden_size: the output neuron number of the hidden layer.
          is_training: whether to build the graph in training mode.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.dbof_cluster_size
        hidden1_size = hidden_size or FLAGS.dbof_hidden_size
        act_fn = self.ACT_FN_MAP.get(FLAGS.dbof_activation)
        assert act_fn is not None, ("dbof_activation is not valid: %s." %
                                    FLAGS.dbof_activation)

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        tf.compat.v1.summary.histogram("input_hist", reshaped_input)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(reshaped_input,
                                             center=True,
                                             scale=True,
                                             is_training=is_training,
                                             scope="input_bn")

        cluster_weights = tf.compat.v1.get_variable(
            "cluster_weights", [feature_size, cluster_size],
            initializer=tf.random_normal_initializer(stddev=1 /
                                                            math.sqrt(feature_size)))
        tf.compat.v1.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        if add_batch_norm:
            activation = slim.batch_norm(activation,
                                         center=True,
                                         scale=True,
                                         is_training=is_training,
                                         scope="cluster_bn")
        else:
            cluster_biases = tf.compat.v1.get_variable(
                "cluster_biases", [cluster_size],
                initializer=tf.random_normal_initializer(stddev=1 /
                                                                math.sqrt(feature_size)))
            tf.compat.v1.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases
        activation = act_fn(activation)
        tf.compat.v1.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])
        activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

        hidden1_weights = tf.compat.v1.get_variable(
            "hidden1_weights", [cluster_size, hidden1_size],
            initializer=tf.random_normal_initializer(stddev=1 /
                                                            math.sqrt(cluster_size)))
        tf.compat.v1.summary.histogram("hidden1_weights", hidden1_weights)
        activation = tf.matmul(activation, hidden1_weights)
        if add_batch_norm:
            activation = slim.batch_norm(activation,
                                         center=True,
                                         scale=True,
                                         is_training=is_training,
                                         scope="hidden1_bn")
        else:
            hidden1_biases = tf.compat.v1.get_variable(
                "hidden1_biases", [hidden1_size],
                initializer=tf.random_normal_initializer(stddev=0.01))
            tf.compat.v1.summary.histogram("hidden1_biases", hidden1_biases)
            activation += hidden1_biases
        activation = act_fn(activation)
        tf.compat.v1.summary.histogram("hidden1_output", activation)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(model_input=activation,
                                               vocab_size=vocab_size,
                                               **unused_params)


class LstmModel(models.BaseModel):
    """Creates a model which uses a stack of LSTMs to represent the video."""

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """See base class.

        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
            input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
            frames for each video (before padding).

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers

        stacked_lstm = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
            for _ in range(number_of_layers)
        ])

        _, state = tf.nn.dynamic_rnn(stacked_lstm,
                                     model_input,
                                     sequence_length=num_frames,
                                     dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(model_input=state[-1].h,
                                               vocab_size=vocab_size,
                                               **unused_params)


class NetVLAD():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        """
        b, n, c1
        """
        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))
        # [c1, K]
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)  # b, n, K

        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [self.cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])  # b, n, K

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, self.feature_size, self.cluster_size],
                                           initializer=tf.random_normal_initializer(
                                               stddev=1 / math.sqrt(self.feature_size)))
        # c1, K
        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)

        return vlad


class NetVLADModelLF(models.BaseModel):
    """Creates a NetVLAD based model.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.netvlad_cluster_size
        hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
        relu = FLAGS.netvlad_relu
        gating = FLAGS.gating
        remove_diag = FLAGS.gating_remove_diag

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)

        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_NetVLAD = NetVLAD(1024, max_frames, cluster_size, add_batch_norm, is_training)
        audio_NetVLAD = NetVLAD(128, max_frames, cluster_size / 2, add_batch_norm, is_training)

        if add_batch_norm:  # and not lightvlad:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_VLAD"):
            vlad_video = video_NetVLAD.forward(reshaped_input[:, 0:1024])

        with tf.variable_scope("audio_VLAD"):
            vlad_audio = audio_NetVLAD.forward(reshaped_input[:, 1024:])

        vlad = tf.concat([vlad_video, vlad_audio], 1)

        vlad_dim = vlad.get_shape().as_list()[1]
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [vlad_dim, hidden1_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

        activation = tf.matmul(vlad, hidden1_weights)

        if add_batch_norm and relu:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="hidden1_bn")

        else:
            hidden1_biases = tf.get_variable("hidden1_biases",
                                             [hidden1_size],
                                             initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram("hidden1_biases", hidden1_biases)
            activation += hidden1_biases

        if relu:
            activation = tf.nn.relu6(activation)

        if gating:
            gating_weights = tf.get_variable("gating_weights_2",
                                             [hidden1_size, hidden1_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(hidden1_size)))

            gates = tf.matmul(activation, gating_weights)

            if remove_diag:
                # removes diagonals coefficients
                diagonals = tf.matrix_diag_part(gating_weights)
                gates = gates - tf.multiply(diagonals, activation)

            if add_batch_norm:
                gates = slim.batch_norm(
                    gates,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="gating_bn")
            else:
                gating_biases = tf.get_variable("gating_biases",
                                                [cluster_size],
                                                initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
                gates += gating_biases

            gates = tf.sigmoid(gates)

            activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


class NetFV():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        covar_weights = tf.get_variable("covar_weights",
                                        [self.feature_size, self.cluster_size],
                                        initializer=tf.random_normal_initializer(mean=1.0, stddev=1 / math.sqrt(
                                            self.feature_size)))

        covar_weights = tf.square(covar_weights)
        eps = tf.constant([1e-6])
        covar_weights = tf.add(covar_weights, eps)

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [self.cluster_size],
                                             initializer=tf.random_normal(stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        if not FLAGS.fv_couple_weights:
            cluster_weights2 = tf.get_variable("cluster_weights2",
                                               [1, self.feature_size, self.cluster_size],
                                               initializer=tf.random_normal_initializer(
                                                   stddev=1 / math.sqrt(self.feature_size)))
        else:
            cluster_weights2 = tf.scalar_mul(FLAGS.fv_coupling_factor, cluster_weights)

        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
        fv1 = tf.matmul(activation, reshaped_input)

        fv1 = tf.transpose(fv1, perm=[0, 2, 1])

        # computing second order FV
        a2 = tf.multiply(a_sum, tf.square(cluster_weights2))

        b2 = tf.multiply(fv1, cluster_weights2)
        fv2 = tf.matmul(activation, tf.square(reshaped_input))

        fv2 = tf.transpose(fv2, perm=[0, 2, 1])
        fv2 = tf.add_n([a2, fv2, tf.scalar_mul(-2, b2)])

        fv2 = tf.divide(fv2, tf.square(covar_weights))
        fv2 = tf.subtract(fv2, a_sum)

        fv2 = tf.reshape(fv2, [-1, self.cluster_size * self.feature_size])

        fv2 = tf.nn.l2_normalize(fv2, 1)
        fv2 = tf.reshape(fv2, [-1, self.cluster_size * self.feature_size])
        fv2 = tf.nn.l2_normalize(fv2, 1)

        fv1 = tf.subtract(fv1, a)
        fv1 = tf.divide(fv1, covar_weights)

        fv1 = tf.nn.l2_normalize(fv1, 1)
        fv1 = tf.reshape(fv1, [-1, self.cluster_size * self.feature_size])
        fv1 = tf.nn.l2_normalize(fv1, 1)

        return tf.concat([fv1, fv2], 1)



class NetFVModelLF(models.BaseModel):
    """Creates a NetFV based model.
       It emulates a Gaussian Mixture Fisher Vector pooling operations

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.fv_cluster_size
        hidden1_size = hidden_size or FLAGS.fv_hidden_size
        relu = FLAGS.fv_relu
        gating = FLAGS.gating

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        tf.summary.histogram("input_hist", reshaped_input)

        video_NetFV = NetFV(1024, max_frames, cluster_size, add_batch_norm, is_training)
        audio_NetFV = NetFV(128, max_frames, cluster_size / 2, add_batch_norm, is_training)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_FV"):
            fv_video = video_NetFV.forward(reshaped_input[:, 0:1024])

        with tf.variable_scope("audio_FV"):
            fv_audio = audio_NetFV.forward(reshaped_input[:, 1024:])

        fv = tf.concat([fv_video, fv_audio], 1)

        fv_dim = fv.get_shape().as_list()[1]
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [fv_dim, hidden1_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

        activation = tf.matmul(fv, hidden1_weights)

        if add_batch_norm and relu:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="hidden1_bn")
        else:
            hidden1_biases = tf.get_variable("hidden1_biases",
                                             [hidden1_size],
                                             initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram("hidden1_biases", hidden1_biases)
            activation += hidden1_biases

        if relu:
            activation = tf.nn.relu6(activation)

        if gating:
            gating_weights = tf.get_variable("gating_weights_2",
                                             [hidden1_size, hidden1_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(hidden1_size)))

            gates = tf.matmul(activation, gating_weights)

            if add_batch_norm:
                gates = slim.batch_norm(
                    gates,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="gating_bn")
            else:
                gating_biases = tf.get_variable("gating_biases",
                                                [cluster_size],
                                                initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
                gates += gating_biases

            gates = tf.sigmoid(gates)

            activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)




class Transformer(models.BaseModel):
    def create_model(self, model_input, **unused_params):
        pass


class ConvModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        feature_size = model_input.get_shape().as_list()[2]
        out1 = slim.convolution(model_input, num_outputs=feature_size, kernel_size=2)
        out2 = slim.convolution(out1, num_outputs=feature_size, kernel_size=2)
        max_pooled = tf.reduce_sum(out2, axis=[1])
        output = slim.fully_connected(max_pooled,
                                      vocab_size,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_regularizer=slim.l2_regularizer(1e-8))
        return {'predictions': output}


def weight_variable(shape, w=0.1):
    initial = tf.truncated_normal(shape, stddev=w)
    return tf.Variable(initial)


def bias_variable(shape, w=0.1):
    initial = tf.constant(w, shape=shape)
    return tf.Variable(initial)