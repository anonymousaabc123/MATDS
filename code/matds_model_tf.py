#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
from keras import backend as K
from keras.layers.core import Layer
from keras.regularizers import l2, L1L2
from keras.initializers import he_normal, Zeros, Ones, TruncatedNormal, glorot_normal, glorot_uniform
from keras.layers import BatchNormalization, Activation, Dropout, Lambda, Concatenate, Dense, Multiply, Conv2D, Flatten, merge
import tensorflow as tf
import logging
from keras.layers import Dense, Concatenate, Activation
from keras.models import Model
import os
import sys
import numpy as np
import time


class MATDS_DeepandCrossModel():
    """
    """

    def __init__(self, config):
        super(MATDS_DeepandCrossModel, self).__init__(config)
        self.n_treatment = int(config.model_def.n_treatment)
        self.negative_sample_weight = float(config.model_def.get('negative_sample_weight', 1))
        self.loss_fn = PolarizedOrdinalLoss(
            self.n_treatment,
            negative_sample_weight=self.negative_sample_weight
        )

    def build_model(self, inputs, labels):
        layerparser = LayerParser(self.config, inputs)

        # Load Params for MLP Layer
        hidden_units = self.config.model_def.get('deep_layers_dim', [16, 8])
        act_fn = self.config.model_def.get('act_fn', 'relu')
        l2_reg = self.config.model_def.get('l2_reg', 0.001)
        keep_prob = self.config.model_def.get('keep_prob', 0.9)
        use_bn = self.config.model_def.get('use_bn', False)
        momentum = self.config.model_def.get('batch_norm_momentum', 0.99)
        use_dropout = self.config.model_def.get('use_dropout', True)
        # fixed_dim = self.config.model_def.get('fixed_dim', None)

        input_lists = []
        input_embed = []
        # deep feature
        deep_feature_input, deep_feature_input_emb = layerparser.get_layer0((self.config.x[0].feature_name, self.config.x[0].embedding_dim))
        add(input_lists, deep_feature_input)
        add(input_embed, deep_feature_input_emb)

        # t_input, t_input_emb = layerparser.get_layer0((self.config.x[1].feature_name, self.config.x[1].embedding_dim))
        # t_input = inputs['treatment']
        # add(input_lists, t_input)
        # add(input_embed, t_input_emb)

        # concatenate dense and deep features
        dnn_input = ListToTensor(input_embed)
        # deep part
        dnn_output_t = MLPLayer(hidden_units, act_fn, l2_reg, keep_prob, use_bn, momentum, use_dropout)(dnn_input)
        dnn_output_m = MLPLayer(hidden_units, act_fn, l2_reg, keep_prob, use_bn, momentum, use_dropout)(dnn_input)
        dnn_output_y = MLPLayer(hidden_units, act_fn, l2_reg, keep_prob, use_bn, momentum, use_dropout)(dnn_input)

        # cross part
        n_layers = self.config.model_def.get('n_layers', 2)
        cross_out_t = CrossLayer(n_layers, l2_reg)(dnn_input)
        cross_out_m = CrossLayer(n_layers, l2_reg)(dnn_input)
        cross_out_y = CrossLayer(n_layers, l2_reg)(dnn_input)

        # concat deep network part with sparse features
        combined_input_t = ListToTensor([dnn_output_t, cross_out_t])  # 只和treatment相关
        combined_input_m = ListToTensor([dnn_output_m, cross_out_m])  # 和treatment,outcome都相关
        combined_input_y = ListToTensor([dnn_output_y, cross_out_y])  # 只和outcome相关

        logging.info('combined_input_t shape {}'.format(combined_input_t.shape))  # shape (?, 128)

        input_t = tf.concat([combined_input_t, combined_input_m], 1)
        input_y = tf.concat([combined_input_y, combined_input_m], 1)

        logging.info('input_t shape {}'.format(input_t.shape))  # input_t shape (?, 256)
        logging.info('input_y shape {}'.format(input_y.shape))  # input_y shape (?, 256)

        disent_units = self.n_treatment
        treatment = inputs['treatment']
        logging.info('treatment: {}'.format(treatment))  # shape=(?, 1), dtype=int32

        logit_m_x_1 = Dense(1, activation=None, use_bias=True, name='mx_function')(input_y)
        m_x_1 = tf.nn.sigmoid(logit_m_x_1)

        # yr-reweight, t是treatment, 其中p_t是提供的超参，pi_0可以是超参，可以是计算出来的
        # t = treatment
        # sigma = tf.nn.sigmoid(Dense(1, activation=None, use_bias=True, name='sigma_function')(combined_input_m))
        # pi_0 = tf.multiply(t, sigma) + tf.multiply(1.0-t, 1.0-sigma)
        # w_t = t / (2*p_t)
        # w_c = (1-t) / (2*(1-p_t))
        # sample_weight = 1 * (1 + (1-self.pi_0)/self.pi_0 * (p_t/(1-p_t))**(2*t-1)) * (w_t+w_c)

        # treatment是一维的，h_t是one_hot表征向量
        h_t = tf.one_hot(tf.reshape(treatment, [-1]), disent_units)  # [batch, disent_units]
        propensity_feature = Dense(disent_units, activation=None, use_bias=True, name='propensity_feature')(input_t)
        g_x = Dense(disent_units, activation=None, use_bias=True, name='g_x')(input_t)

        logit_m_x_2 = tf.reduce_mean(g_x * propensity_feature, axis=-1, keepdims=True)
        logit_gh = tf.reduce_mean(g_x * h_t, axis=-1, keepdims=True)
        y_r = m_x_1 + tf.reduce_mean(g_x * (h_t - propensity_feature), axis=-1, keepdims=True)

        # loss
        label = tf.cast(labels['label'], tf.float32)
        loss_m_x_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit_m_x_1))
        loss_m_x_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit_m_x_2))
        loss_pf = tf.reduce_mean(tf.squared_difference(h_t, propensity_feature))
        loss_r = tf.reduce_mean(tf.squared_difference(label, y_r))
        loss_gh = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit_gh))

        # imblance loss
        p_ipm = 0.5
        sigma = 0.25
        # loss_imb = self.calculate_disc_mmd2_lin(combined_input_y, treatment, p_ipm)
        loss_imb = tf.abs(self.calculate_disc_mmd2_rbf(combined_input_y, treatment, p_ipm, sigma))
        logging.info('loss_imb: {}'.format(loss_imb))

        # self.loss = (loss_pf + loss_r) / 2
        # self.loss = loss_m_x_1 + 0.1 * loss_pf + loss_r
        # self.loss = loss_m_x_1 + 0.1 * loss_pf + loss_r + loss_imb
        self.loss = (loss_m_x_1 + loss_pf + loss_r + loss_imb) / 4
        prediction_score = logit_gh

        # metrics
        self.metrics = {'loss_print': self.loss,
                        'loss_m_x_1': loss_m_x_1,
                        'loss_m_x_2': loss_m_x_2,
                        'loss_pf': loss_pf,
                        'loss_r': loss_r,
                        'loss_gh': loss_gh}

        self.prediction_result = {
            'prediction': prediction_score,
            'm_x_1': tf.sigmoid(logit_m_x_1),
            'm_x_2': tf.sigmoid(logit_m_x_2),
            'y_r': y_r,
            'propensity_feature': propensity_feature,
            'loss_print': self.loss
        }

        tf.summary.scalar('loss_m_x_1', loss_m_x_1)
        tf.summary.scalar('loss_imb', loss_imb)
        tf.summary.scalar('loss_pf', loss_pf)
        tf.summary.scalar('loss_r', loss_r)
        tf.summary.scalar('loss', self.loss)

        return self.prediction_result

    def calculate_disc_mmd2_lin(self, X, t, p):
        '''
        mmd2_lin
        '''
        it = tf.where(t > 0)[:, 0]
        ic = tf.where(t < 1)[:, 0]

        Xc = tf.gather(X, ic)  # shape=(?, 128)
        Xt = tf.gather(X, it)  # shape=(?, 128)

        mean_control = tf.reduce_mean(Xc)
        mean_treated = tf.reduce_mean(Xt)

        mmd = tf.reduce_sum(tf.square(2.0 * p * mean_treated - 2.0 * (1.0 - p) * mean_control))

        return mmd * 1e6

    def calculate_disc_mmd2_rbf(self, X, t, p, sig):
        """ Computes the l2-RBF MMD for X given t """

        it = tf.where(t > 0)[:, 0]
        ic = tf.where(t < 1)[:, 0]

        Xc = tf.gather(X, ic)
        Xt = tf.gather(X, it)

        Kcc = tf.exp(-self.pdist2sq(Xc, Xc) / tf.square(sig))
        Kct = tf.exp(-self.pdist2sq(Xc, Xt) / tf.square(sig))
        Ktt = tf.exp(-self.pdist2sq(Xt, Xt) / tf.square(sig))

        m = tf.to_float(tf.shape(Xc)[0])
        n = tf.to_float(tf.shape(Xt)[0])

        mmd = tf.square(1.0 - p) / (m * (m - 1.0)) * (tf.reduce_sum(Kcc) - m)
        mmd = mmd + tf.square(p) / (n * (n - 1.0)) * (tf.reduce_sum(Ktt) - n)
        mmd = mmd - 2.0 * p * (1.0 - p) / (m * n) * tf.reduce_sum(Kct)
        mmd = 4.0 * mmd

        return mmd * 1e1

    def pdist2sq(self, X, Y):
        """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
        C = -2 * tf.matmul(X, tf.transpose(Y))
        nx = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
        ny = tf.reduce_sum(tf.square(Y), 1, keep_dims=True)
        D = (C + tf.transpose(ny)) + nx
        return D

    def get_prediction_result(self, **kwargs):
        return self.prediction_result

    def get_loss(self, **kwargs):
        return self.loss

    def get_metrics(self, **kwargs):
        return self.metrics

    def get_summary_op(self):
        return tf.summary.merge_all(), None


class PolarizedOrdinalLoss:
    def __init__(self, n_treatment=3, negative_sample_weight=1):
        '''
        negative_sample_weight is an extra coefficient for negative sample loss, default 1.
        '''
        assert n_treatment > 1, "n_treatment is required to be at least 2."
        self.n_treatment = n_treatment
        self.negative_sample_weight = negative_sample_weight
        self.threshold = tf.Variable(
            tf.linspace(-(n_treatment + 1) / 2, (n_treatment + 1) / 2, n_treatment),
            trainable=True,
            name="threshold",
        )

    def __call__(self, sample_list, model_output):
        assert "labels" in sample_list, "sample_list should contain labels."
        label = sample_list["labels"]
        assert "logits" in model_output, "model_output should contain logits."
        input = model_output["logits"]  # input and treatment should be of the same size

        input = tf.reshape(input, [-1, 1])
        label = tf.reshape(label, [-1])
        treatment = tf.cast(tf.abs(label), tf.int32)
        label = tf.sign(tf.cast(label, tf.float32))  # label in {1, -1}
        batch_size = tf.reduce_sum(tf.ones_like(treatment))

        # epsilon=1-(input-threshold)*label
        epsilon = 1.0 - (input - self.threshold) * tf.expand_dims(label, -1)
        # Set the self.n_treatment-treatment-1 location to 1
        scatter_rows = tf.range(batch_size)
        treatment_scatter = tf.scatter_nd(
            indices=tf.transpose(tf.stack([scatter_rows, treatment - 1])),
            updates=tf.ones_like(treatment, dtype=tf.float32),
            shape=[batch_size, self.n_treatment],
            name="treatment_scatter",
        )
        mask_positive = tf.cumsum(treatment_scatter, axis=-1, exclusive=False, reverse=True)
        mask_negative = tf.cumsum(treatment_scatter, axis=-1, exclusive=False, reverse=False)
        # Select positive or negative mask according to the label.
        # Extract one row for each sample
        mask = tf.where(
            label > 0,
            mask_positive,
            mask_negative * self.negative_sample_weight
        )
        loss = tf.math.maximum(epsilon * mask, 0)  # Hinge loss

        return tf.reduce_mean(loss)

    def __str__(self):
        """Print out the threshold values"""
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            threshold = self.threshold.eval()
        return f"ordinal thresholds = {threshold}"

    def pred(self, input):
        """Convert input values to treatment via thresholding
        - Input: function values.
        - Output: int, proposed treatment, [0, n_treatment].
        """
        return tf.reduce_sum(
            tf.cast((tf.expand_dims(input, -1) - self.threshold) > 0, tf.int32), axis=-1
        )


def ListToTensor(inputs, fixed_dim=None, apply_dropout=False, keep_prob=0.2, require_3d_output=False):
    """
    convert list of tensors to 3d tensors with shape (batch, fields, dim)
    or 2d tensor with shape (batch, fields*dim)

    Args:
        inputs: list, each element with shape (batch, dim), ie: embedding layer from inputs
    return:
        tensor: (batch, fields, dim) if fixed_dim is not None, otherwise, (batch, fields*dim)
    """
    if inputs is None or (not isinstance(inputs, (list, tuple))):
        # inputs is None or a tensor, then there is nothing to do
        logging.info('expect non-empty list, got {}'.format(type(inputs)))
        return inputs

    if fixed_dim is None and not require_3d_output:
        # hereby, inputs is a list of 2d tensor
        return Concatenate(axis=-1)(inputs) if len(inputs) > 1 else inputs[0]

    field_size = len(inputs)
    inputs_expand = []
    for i in range(field_size):
        emb = Lambda(lambda x: K.expand_dims(x, axis=1))(inputs[i])
        if apply_dropout:
            emb = Dropout(rate=keep_prob)(Activation(activation='relu')(emb))
        if fixed_dim:
            emb = Dense(fixed_dim)(emb)
        inputs_expand.append(emb)

    return Concatenate(axis=1)(inputs_expand) if len(inputs_expand) > 1 else inputs_expand[0]


class MLPLayer(Layer):
    """
    MLP layers
    inputs: 2d tensor (batch, dim)
    output: 2d tensor (batch, hidden_units[-1])
    """

    def __init__(self,
                 hidden_units=[16, 8, 4],
                 act_fn='relu',
                 l2_reg=0.001,
                 keep_prob=0.3,
                 use_bn=False,
                 momentum=0.9,
                 use_dropout=False,
                 **kwargs):
        """
        Args:
            hidden_units: int or list of int, number of units in each layer
            act_fn: str or list of str, activation function in each layer
            l2_reg: float or list of float, [0,1], L2 Regularization strength applied to kernel weights matrix
            keep_prob: float or list of float, [0, 1], probability of dropout
            use_bn: bool or list of bool, if True apply BatchNormalization in each layer
            momentum: float, momentum in BatchNormalization
            use_dropout: bool, if True, apply dropout after each layer
        """
        super(MLPLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.act_fn = act_fn
        self.l2_reg = l2_reg
        self.keep_prob = keep_prob
        self.use_bn = use_bn
        self.momentum = momentum
        self.use_dropout = use_dropout
        self._expand_dim()

    def _expand_dim(self):
        if isinstance(self.hidden_units, list):
            if isinstance(self.act_fn, list) and len(self.act_fn) != len(self.hidden_units):
                raise ValueError("hidden_units and activation list must have the same lenght, got {}{}".format(
                    len(self.hidden_units), len(self.act_fn)))
            self.act_fn = [self.act_fn for _ in range(len(self.hidden_units))]

            if isinstance(self.l2_reg, list) and len(self.l2_reg) != len(self.hidden_units):
                raise ValueError('hidden_units and l2_reg must have the same lenght, got {}{}'.format(
                    len(self.hidden_units), len(self.l2_reg)))
            self.l2_reg = [self.l2_reg for _ in range(len(self.hidden_units))]

            if isinstance(self.keep_prob, list) and len(self.keep_prob) != len(self.hidden_units):
                raise ValueError('hidden_units and keep_prob must have the same length, got {}{}'.format(
                    len(self.hidden_units), len(self.keep_prob)))
            self.keep_prob = [self.keep_prob for _ in range(len(self.hidden_units))]

            if isinstance(self.use_bn, list) and len(self.hidden_units) != len(self.use_bn):
                raise ValueError('hidden_units and use_bn must have the same length, got {}{}'.format(
                    len(self.hidden_units), len(self.use_bn)))
            self.use_bn = [self.use_bn for _ in range(len(self.hidden_units))]
            self.use_dropout = [self.use_dropout for _ in range(len(self.hidden_units))]

    def build(self, input_shape):
        """
        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).
        """
        input_size = int(input_shape[-1])
        hidden_size = [int(input_size)] + list(self.hidden_units)
        self.kernels, self.bias = [], []
        for i in range(len(hidden_size) - 1):
            self.kernels.append(
                self.add_weight(
                    name="kernel_{}".format(i),
                    shape=(hidden_size[i], hidden_size[i + 1]),
                    initializer=he_normal(),
                    regularizer=L1L2(0, self.l2_reg[i]),
                    trainable=True))
            self.bias.append(
                self.add_weight(
                    name='bias_{}'.format(i), shape=(hidden_size[i + 1],), initializer=Zeros(), trainable=True))

        assert (len(list(self.hidden_units)) == len(self.kernels)), ('hidden_units and n_layers must be the same')
        super(MLPLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        """
        Args:
            inputs: Input tensor, or list/tuple of input tensors.
            training: bool
        Returns:
           inputs: 2d tensor
        """
        deep_inputs = inputs
        for i in range(len(self.hidden_units)):
            # To Do: Test the units here, xw_plus_b only works for 2d tensor
            logging.info('MLP layer inputs shape: {}'.format(inputs.shape))
            logging.info('kernel {} shape {}'.format(i, self.kernels[i].shape))

            fc = K.dot(deep_inputs, self.kernels[i])
            fc = K.bias_add(fc, self.bias[i])
            logging.info('MLP call fc shape {}'.format(fc.shape))
            if self.use_bn[i]:
                fc = BatchNormalization(momentum=self.momentum)(fc)

            # fc = get_activation_layer(self.act_fn[i], fc)
            fc = Activation(self.act_fn[i])(fc)

            logging.info('fc after activation shape: {}'.format(fc.shape))
            if self.use_dropout[i]:
                fc = Dropout(rate=1 - self.keep_prob[i])(fc, )
            deep_inputs = fc
            logging.info('deep_inputs in MLP layer {}'.format(deep_inputs.shape))
        logging.info('done call in deep_input MLP layer {}'.format(deep_inputs.shape))
        return deep_inputs

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
            logging.info('MLP layer output shape {}'.format(shape))
        else:
            shape = input_shape
        return tuple(shape)

    def get_config(self):
        config = {
            'hidden_units': self.hidden_units,
            'act_fn': self.act_fn,
            'l2_reg': self.l2_reg,
            'keep_prob': self.keep_prob,
            'use_bn': self.use_bn
        }
        base_config = super(MLPLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CrossLayer(Layer):
    """
    CrossLayer
    paper: Deep & Cross Network for Ad Click Predictions

    X_{l+1} = X_0 * X_l^T * W_l + b_l + X_l

    note: X_l (batch, n_dim)
          W_l (batch, )
          b_l (batch, )

    inputs: 2d tensor (batch_size, n_dim)
    outputs: 2d tensor (batch_size, n_dim)
    """

    def __init__(self, n_layers=2, l2_reg=0.001, **kwargs):
        """
        Args:
            n_layers: int, number of layers
            l2_reg: float
        """
        super(CrossLayer, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.l2_reg = l2_reg
        logging.info('cross layer, n_layers {}, l2_reg {}'.format(n_layers, l2_reg))

    def build(self, input_shape):
        """
        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).
        """
        input_size = int(input_shape[-1])
        logging.info('input_size {}'.format(input_size))
        self.kernels, self.bias = [], []
        for i in range(self.n_layers):
            self.kernels.append(
                self.add_weight(
                    name='weight_{}'.format(i),
                    shape=(input_size, 1),
                    initializer=he_normal(),
                    # initializer='glorot_uniform',
                    regularizer=L1L2(0, self.l2_reg),
                    trainable=True))
            self.bias.append(
                self.add_weight(name='bias_{}'.format(i), shape=(input_size,), initializer=Zeros(), trainable=True))
        super(CrossLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: 2d tensor (batch_size, n_dim)
        returns:
            2d tensor (batch_size, n_dim)
        """
        inputs = K.expand_dims(inputs, axis=1)  # X_0
        deep_inputs = inputs
        # deep_inputs (batch, 1, d)
        logging.info('Cross Layer deep_inputs {}'.format(deep_inputs))
        for i in range(self.n_layers):
            logging.info('kernel size {}'.format(self.kernels[i]))
            deep_inputs = K.batch_dot(K.dot(deep_inputs, self.kernels[i]), inputs, axes=[1, 1]) + deep_inputs
            deep_inputs = K.bias_add(deep_inputs, self.bias[i])
            logging.info('Cross Layer inside deep_inputs {}'.format(deep_inputs))

        deep_inputs = K.squeeze(deep_inputs, axis=1)
        return deep_inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'n_layers': self.n_layers, 'l2_reg': self.l2_reg}
        base_config = super(CrossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
