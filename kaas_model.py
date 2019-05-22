#!/usr/bin/env python
# coding: utf-8
# @Author: FS-J

"""

This module is KAAS deep nurual network models(i.e., LSTM in our paper and CNN for choice.).

"""

import tensorflow as tf

from model import APModel



class KAAS_biLSTM(KAASModel):
    """This class implements KAAS-biLSTM model."""

    def _encode(self, x, length):
        params = self.params
        fw_cells = []
        bw_cells = []
        for _ in range(params.num_layers):
            fw = tf.nn.rnn_cell.LSTMCell(params.hidden_units)
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                fw = tf.nn.rnn_cell.DropoutWrapper(fw, output_keep_prob=params.dropout)
            fw_cells.append(fw)
            bw = tf.nn.rnn_cell.LSTMCell(params.hidden_units)
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                bw = tf.nn.rnn_cell.DropoutWrapper(bw, output_keep_prob=params.dropout)
            bw_cells.append(bw)
        cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=fw_cells)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=bw_cells)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x, sequence_length=length, dtype=tf.float32)
        outputs = tf.concat([output_fw, output_bw], axis=-1, name="output")  # [batch_size, max_time, output_size]

        return outputs

    

class KAAS_CNN(KAASModel):
    """This class implements KAAS-CNN model."""

    def _encode(self, x, length):
        params = self.params

        # Use tf high level API tf.layers
        pooled_outputs = []
        for i, filter_size in enumerate(map(int, params.filter_sizes.split(','))):
            with tf.variable_scope("conv"):
                conv = tf.layers.conv1d(
                    x, params.num_filters, filter_size,
                    padding="same",
                    bias_initializer=tf.constant_initializer(0.1),
                    name="filter_size_%d" % filter_size)  # (batch_size, seq_length，num_filters)
                pooled_outputs.append(conv)
        # Combine all the features
        outputs = tf.concat(pooled_outputs, 2, name="output")  # (batch_size, seq_length, num_filters_total)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            outputs = tf.nn.dropout(outputs, params.dropout, name="output")

        return outputs


