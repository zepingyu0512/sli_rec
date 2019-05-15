import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn_cell_impl import  _Linear
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from keras import backend as K
import os
import sys
import random
import tempfile
from subprocess import call

class Time4AILSTMCell(RNNCell):

  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None):

    super(Time4AILSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units
    self._linear1 = None
    self._linear2 = None
    self._time_input_w1 = None
    self._time_input_w2 = None
    self._time_kernel_w1 = None
    self._time_kernel_t1 = None
    self._time_bias1 = None
    self._time_kernel_w2 = None
    self._time_kernel_t2 = None
    self._time_bias2 = None
    self._o_kernel_t1 = None
    self._o_kernel_t2 = None
    if self._use_peepholes:
      self._w_f_diag = None
      self._w_i_diag = None
      self._w_o_diag = None

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size
    
  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)
      
  def call(self, inputs, state, att_score=None):
    time_now_score = tf.expand_dims(inputs[:,-1], -1)
    time_last_score = tf.expand_dims(inputs[:,-2], -1)
    inputs = inputs[:,:-2]
    inputs = inputs * att_score
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
            
    if self._time_kernel_w1 is None:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        with vs.variable_scope(unit_scope):
          self._time_input_w1 = vs.get_variable(
              "_time_input_w1", shape=[self._num_units], dtype=dtype)
          self._time_input_bias1 = vs.get_variable(
              "_time_input_bias1", shape=[self._num_units], dtype=dtype)
          self._time_input_w2 = vs.get_variable(
              "_time_input_w2", shape=[self._num_units], dtype=dtype)
          self._time_input_bias2 = vs.get_variable(
              "_time_input_bias2", shape=[self._num_units], dtype=dtype)
          self._time_kernel_w1 = vs.get_variable(
              "_time_kernel_w1", shape=[input_size, self._num_units], dtype=dtype)
          self._time_kernel_t1 = vs.get_variable(
              "_time_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)
          self._time_bias1 = vs.get_variable(
              "_time_bias1", shape=[self._num_units], dtype=dtype)
          self._time_kernel_w2 = vs.get_variable(
              "_time_kernel_w2", shape=[input_size, self._num_units], dtype=dtype)
          self._time_kernel_t2 = vs.get_variable(
              "_time_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)
          self._time_bias2 = vs.get_variable(
              "_time_bias2", shape=[self._num_units], dtype=dtype)
          self._o_kernel_t1 = vs.get_variable(
              "_o_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)    
          self._o_kernel_t2 = vs.get_variable(
              "_o_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)  
                
    time_now_input = tf.nn.tanh(time_now_score * self._time_input_w1 + self._time_input_bias1)
    time_last_input = tf.nn.tanh(time_last_score * self._time_input_w2 + self._time_input_bias2)      

    time_now_state = math_ops.matmul(inputs, self._time_kernel_w1) + math_ops.matmul(time_now_input, self._time_kernel_t1) + self._time_bias1
    time_last_state = math_ops.matmul(inputs, self._time_kernel_w2) + math_ops.matmul(time_last_input, self._time_kernel_t2) + self._time_bias2
    
    if self._linear1 is None:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        if self._num_unit_shards is not None:
          unit_scope.set_partitioner(
              partitioned_variables.fixed_size_partitioner(
                  self._num_unit_shards))
        self._linear1 = _Linear([inputs, m_prev], 4 * self._num_units, True)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = self._linear1([inputs, m_prev])
    i, j, f, o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)
    o = o + math_ops.matmul(time_now_input, self._o_kernel_t1) + math_ops.matmul(time_last_input, self._o_kernel_t2)   
    # Diagonal connections
    if self._use_peepholes and not self._w_f_diag:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        with vs.variable_scope(unit_scope):
          self._w_f_diag = vs.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          self._w_i_diag = vs.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          self._w_o_diag = vs.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

    if self._use_peepholes:
      c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * sigmoid(time_last_state) * c_prev +
           sigmoid(i + self._w_i_diag * c_prev) * sigmoid(time_now_state) * self._activation(j))
    else:
      c = (sigmoid(f + self._forget_bias) * sigmoid(time_last_state) * c_prev + sigmoid(i) * sigmoid(time_now_state) * self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      if self._linear2 is None:
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope, initializer=self._initializer):
          with vs.variable_scope("projection") as proj_scope:
            if self._num_proj_shards is not None:
              proj_scope.set_partitioner(
                  partitioned_variables.fixed_size_partitioner(
                      self._num_proj_shards))
            self._linear2 = _Linear(m, self._num_proj, False)
      m = self._linear2(m)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state

class Time4ALSTMCell(RNNCell):

  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None):

    super(Time4ALSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units
    self._linear1 = None
    self._linear2 = None
    self._time_input_w1 = None
    self._time_input_w2 = None
    self._time_kernel_w1 = None
    self._time_kernel_t1 = None
    self._time_bias1 = None
    self._time_kernel_w2 = None
    self._time_kernel_t2 = None
    self._time_bias2 = None
    self._o_kernel_t1 = None
    self._o_kernel_t2 = None
    if self._use_peepholes:
      self._w_f_diag = None
      self._w_i_diag = None
      self._w_o_diag = None

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size
    
  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)
      
  def call(self, inputs, state, att_score=None):
    time_now_score = tf.expand_dims(inputs[:,-1], -1)
    time_last_score = tf.expand_dims(inputs[:,-2], -1)
    inputs = inputs[:,:-2]
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
            
    if self._time_kernel_w1 is None:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        with vs.variable_scope(unit_scope):
          self._time_input_w1 = vs.get_variable(
              "_time_input_w1", shape=[self._num_units], dtype=dtype)
          self._time_input_bias1 = vs.get_variable(
              "_time_input_bias1", shape=[self._num_units], dtype=dtype)
          self._time_input_w2 = vs.get_variable(
              "_time_input_w2", shape=[self._num_units], dtype=dtype)
          self._time_input_bias2 = vs.get_variable(
              "_time_input_bias2", shape=[self._num_units], dtype=dtype)
          self._time_kernel_w1 = vs.get_variable(
              "_time_kernel_w1", shape=[input_size, self._num_units], dtype=dtype)
          self._time_kernel_t1 = vs.get_variable(
              "_time_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)
          self._time_bias1 = vs.get_variable(
              "_time_bias1", shape=[self._num_units], dtype=dtype)
          self._time_kernel_w2 = vs.get_variable(
              "_time_kernel_w2", shape=[input_size, self._num_units], dtype=dtype)
          self._time_kernel_t2 = vs.get_variable(
              "_time_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)
          self._time_bias2 = vs.get_variable(
              "_time_bias2", shape=[self._num_units], dtype=dtype)
          self._o_kernel_t1 = vs.get_variable(
              "_o_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)    
          self._o_kernel_t2 = vs.get_variable(
              "_o_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)  
                
    time_now_input = tf.nn.tanh(time_now_score * self._time_input_w1 + self._time_input_bias1)
    time_last_input = tf.nn.tanh(time_last_score * self._time_input_w2 + self._time_input_bias2)      

    time_now_state = math_ops.matmul(inputs, self._time_kernel_w1) + math_ops.matmul(time_now_input, self._time_kernel_t1) + self._time_bias1
    time_last_state = math_ops.matmul(inputs, self._time_kernel_w2) + math_ops.matmul(time_last_input, self._time_kernel_t2) + self._time_bias2
    
    if self._linear1 is None:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        if self._num_unit_shards is not None:
          unit_scope.set_partitioner(
              partitioned_variables.fixed_size_partitioner(
                  self._num_unit_shards))
        self._linear1 = _Linear([inputs, m_prev], 4 * self._num_units, True)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = self._linear1([inputs, m_prev])
    i, j, f, o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)
    o = o + math_ops.matmul(time_now_input, self._o_kernel_t1) + math_ops.matmul(time_last_input, self._o_kernel_t2)   
    # Diagonal connections
    if self._use_peepholes and not self._w_f_diag:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        with vs.variable_scope(unit_scope):
          self._w_f_diag = vs.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          self._w_i_diag = vs.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          self._w_o_diag = vs.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

    if self._use_peepholes:
      c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * sigmoid(time_last_state) * c_prev +
           sigmoid(i + self._w_i_diag * c_prev) * sigmoid(time_now_state) * self._activation(j))
    else:
      c = (sigmoid(f + self._forget_bias) * sigmoid(time_last_state) * c_prev + sigmoid(i) * sigmoid(time_now_state) * self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      if self._linear2 is None:
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope, initializer=self._initializer):
          with vs.variable_scope("projection") as proj_scope:
            if self._num_proj_shards is not None:
              proj_scope.set_partitioner(
                  partitioned_variables.fixed_size_partitioner(
                      self._num_proj_shards))
            self._linear2 = _Linear(m, self._num_proj, False)
      m = self._linear2(m)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type
    c = att_score * c + (1. - att_score) * c_prev
    m = att_score * m + (1. - att_score) * m_prev
    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state
    
class VecAttGRUCell(RNNCell):
  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(VecAttGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units
  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)
  def call(self, inputs, state, att_score=None):
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _Linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([inputs, r_state]))
    u = (1.0 - att_score) * u
    new_h = u * state + (1 - u) * c
    return new_h, new_h

def attention_HAN(inputs, attention_size=None, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
        
    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer    
    if attention_size == None:
        attention_size = hidden_size
    
    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
    alphas = tf.reshape(alphas, [-1, tf.shape(inputs)[1]])
    output = inputs * tf.expand_dims(alphas, -1)
    output = tf.reshape(output, tf.shape(inputs))

    if not return_alphas:
        return output
    else:
        return output, alphas
                
def attention_DIN(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        facts = tf.concat(facts, 2)
        print ("querry_size mismatch")
        query = tf.concat(values = [
        query,
        query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output

def attention_FCN(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False, forCnn=False, scope = ""):
    if isinstance(facts, tuple):
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query,scope=scope)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    
    # Mask
    key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output

def dice(_x, axis=-1, epsilon=0.000000001, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha'+name, _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
        input_shape = list(_x.get_shape())
    
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]
  
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    x_p = tf.sigmoid(x_normed)  
    return alphas * (1.0 - x_p) * _x + x_p * _x
    
def prelu(_x, scope=''):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)
        
def calc_auc(raw_arr):
    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc
    
def shuffle(file):
    tf_os, tpath = tempfile.mkstemp(dir='data')
    tf = open(tpath, 'w')

    fd = open(file, "r")
    for l in fd:
        print >> tf, l.strip("\n")
    tf.close()

    lines = open(tpath, 'r').readlines()
    random.shuffle(lines)
    path, filename = os.path.split(os.path.realpath(file))
    fd = tempfile.TemporaryFile(prefix=filename + '.shuf', dir=path)

    for l in lines:
        s = l.strip("\n")
        print >> fd, s

    fd.seek(0)
    os.remove(tpath)

    return fd

def prepare_data(source, target, maxlen = 100):
    sequence_length = [len(s[3]) for s in source]
    item_history = [s[3] for s in source]
    cate_history = [s[4] for s in source]
    timeinterval_history = [s[5] for s in source]
    timelast_history = [s[6] for s in source]
    timenow_history = [s[7] for s in source]

    sequence_length1 = []
    item_history1 = []
    cate_history1 = []
    timeinterval_history1 = []
    timelast_history1 = []
    timenow_history1 = []
    for seqlen, inp in zip(sequence_length, source):
        if seqlen > maxlen:
            item_history1.append(inp[3][seqlen - maxlen:])
            cate_history1.append(inp[4][seqlen - maxlen:])
            timeinterval_history1.append(inp[5][seqlen - maxlen:])
            timelast_history1.append(inp[6][seqlen - maxlen:])
            timenow_history1.append(inp[7][seqlen - maxlen:])
            sequence_length1.append(maxlen)
        else:
            item_history1.append(inp[3])
            cate_history1.append(inp[4])
            timeinterval_history1.append(inp[5])
            timelast_history1.append(inp[6])
            timenow_history1.append(inp[7])
            sequence_length1.append(seqlen)
                
    sequence_length = sequence_length1
    item_history = item_history1
    cate_history = cate_history1
    timeinterval_history = timeinterval_history1
    timelast_history = timelast_history1
    timenow_history = timenow_history1

    if len(sequence_length) < 1:
        return None, None, None, None

    n_samples = len(item_history)
    maxlen_x = np.max(sequence_length)

    item_history_np = np.zeros((n_samples, maxlen_x)).astype('int64')
    cate_history_np = np.zeros((n_samples, maxlen_x)).astype('int64')
    timeinterval_history_np = np.zeros((n_samples, maxlen_x)).astype('float32')
    timelast_history_np = np.zeros((n_samples, maxlen_x)).astype('float32')
    timenow_history_np = np.zeros((n_samples, maxlen_x)).astype('float32')
    mid_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, [s_item, s_cate, s_tint, s_tlast, s_tnow] in enumerate(zip(item_history, cate_history, 
                                                                    timeinterval_history, timelast_history, timenow_history)):
        mid_mask[idx, :sequence_length[idx]] = 1.0
        item_history_np[idx, :sequence_length[idx]] = s_item
        cate_history_np[idx, :sequence_length[idx]] = s_cate
        timeinterval_history_np[idx, :sequence_length[idx]] = s_tint
        timelast_history_np[idx, :sequence_length[idx]] = s_tlast
        timenow_history_np[idx, :sequence_length[idx]] = s_tnow

    user = np.array([inp[0] for inp in source])
    targetitem = np.array([inp[1] for inp in source])
    targetcategory = np.array([inp[2] for inp in source])

    return user, targetitem, targetcategory, item_history_np, cate_history_np, timeinterval_history_np, timelast_history_np, timenow_history_np, mid_mask, np.array(target), np.array(sequence_length)
    
def evaluate_epoch(sess, test_data, model):

    test_loss_sum = 0.0
    test_accuracy_sum = 0.0
    count = 0
    output = []
    for src, tgt in test_data:
        count += 1
        user, targetitem, targetcategory, item_history, cate_history, timeinterval_history, timelast_history, timenow_history, mid_mask, label, seq_len = prepare_data(src, tgt)
        test_prob, test_loss, test_acc = model.calculate(sess, [user, targetitem, targetcategory, item_history, cate_history, timeinterval_history, 
                                                                timelast_history, timenow_history, mid_mask, label, seq_len])
        test_loss_sum += test_loss
        test_accuracy_sum += test_acc
        test_prob_1 = test_prob[:, 0].tolist()
        label_1 = label[:, 0].tolist()
        for p ,t in zip(test_prob_1, label_1):
            output.append([p, t])
    test_auc = calc_auc(output)
    test_loss = test_loss_sum / count
    test_accuracy = test_accuracy_sum / count
    
    return test_auc, test_loss, test_accuracy
